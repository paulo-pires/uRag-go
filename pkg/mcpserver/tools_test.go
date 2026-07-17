package mcpserver

import (
	"context"
	"database/sql"
	"hash/fnv"
	"math"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	_ "modernc.org/sqlite"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"urag-go/pkg/graph"
	"urag-go/pkg/rag"
	urasql "urag-go/pkg/sql"
	"urag-go/pkg/tree"
)

// fakeEmbedding gera um vetor determinístico a partir do hash do texto, mesmo
// padrão usado em pkg/rag/vector_test.go — evita dependência de Ollama real.
func fakeEmbedding(_ context.Context, text string) ([]float32, error) {
	h := fnv.New32a()
	h.Write([]byte(text))
	seed := h.Sum32()

	vec := make([]float32, 8)
	var sumSq float64
	for i := range vec {
		v := float32((seed>>uint(i)&0xFF))/255 + 0.001
		vec[i] = v
		sumSq += float64(v * v)
	}
	norm := float32(math.Sqrt(sumSq))
	for i := range vec {
		vec[i] /= norm
	}
	return vec, nil
}

// fakeGraphExtraction devolve uma extração JSON fixa quando o texto contém
// "Maria", senão devolve um JSON vazio — determinístico, sem Ollama real.
func fakeGraphExtraction(_ context.Context, prompt string) (string, error) {
	if strings.Contains(prompt, "Maria") {
		return `{"entities":[{"name":"Maria","type":"Pessoa"},{"name":"Ignus","type":"Empresa"}],"relations":[{"source":"Maria","target":"Ignus","relation":"trabalha_em"}]}`, nil
	}
	return `{"entities":[],"relations":[]}`, nil
}

// fakeTreeNavigate sempre escolhe o primeiro item, determinístico.
func fakeTreeNavigate(_ context.Context, _ string) (string, error) {
	return "1", nil
}

// fakeSQLGenerate sempre devolve a mesma query SELECT, determinística.
func fakeSQLGenerate(_ context.Context, _ string) (string, error) {
	return "SELECT COUNT(*) as total FROM t", nil
}

func newTestServer(t *testing.T, withSQL bool) *Server {
	t.Helper()

	vector, err := rag.NewWithEmbedding(rag.Config{}, fakeEmbedding)
	if err != nil {
		t.Fatalf("rag.NewWithEmbedding: %v", err)
	}

	s := &Server{
		mcp:    mcp.NewServer(&mcp.Implementation{Name: "test"}, nil),
		vector: vector,
		graph:  graph.NewWithCompletion(fakeGraphExtraction),
		tree:   tree.NewWithNavigator(fakeTreeNavigate),
	}
	if withSQL {
		// DSN de arquivo real (não ":memory:") porque Store.New introspecta e
		// mantém sua própria conexão; ":memory:" isolado por conexão faria a
		// introspecção enxergar um banco vazio diferente do populado aqui.
		dsn := filepath.Join(t.TempDir(), "test.db")
		setupDB, err := sql.Open("sqlite", dsn)
		if err != nil {
			t.Fatalf("abrir banco de setup: %v", err)
		}
		if _, err := setupDB.Exec(`CREATE TABLE t (id INTEGER PRIMARY KEY)`); err != nil {
			t.Fatalf("criar tabela: %v", err)
		}
		if err := setupDB.Close(); err != nil {
			t.Fatalf("fechar banco de setup: %v", err)
		}

		sqlStore, err := urasql.NewWithGenerator(dsn, fakeSQLGenerate)
		if err != nil {
			t.Fatalf("sql.NewWithGenerator: %v", err)
		}
		t.Cleanup(func() { sqlStore.Close() })
		s.sql = sqlStore
	}
	s.registerTools()
	return s
}

func TestRegisterToolsSQLConditional(t *testing.T) {
	withoutSQL := newTestServer(t, false)
	if withoutSQL.sql != nil {
		t.Fatalf("esperava sql nil quando SQLDSN vazio")
	}

	withSQL := newTestServer(t, true)
	if withSQL.sql == nil {
		t.Fatalf("esperava sql configurado")
	}
}

func TestVectorAddThenQuery(t *testing.T) {
	s := newTestServer(t, false)
	ctx := context.Background()

	_, addOut, err := s.vectorAdd(ctx, nil, VectorAddInput{Documents: []DocumentInput{
		{ID: "doc1", Content: "gatos gostam de dormir"},
		{ID: "doc2", Content: "carros precisam de gasolina"},
	}})
	if err != nil {
		t.Fatalf("vectorAdd: %v", err)
	}
	if addOut.Added != 2 {
		t.Fatalf("Added = %d, esperava 2", addOut.Added)
	}

	_, queryOut, err := s.vectorQuery(ctx, nil, VectorQueryInput{Question: "gatos gostam de dormir", TopK: 1})
	if err != nil {
		t.Fatalf("vectorQuery: %v", err)
	}
	if len(queryOut.Results) != 1 || queryOut.Results[0].Document.ID != "doc1" {
		t.Fatalf("resultado inesperado: %+v", queryOut.Results)
	}
}

func TestGraphAddThenQuery(t *testing.T) {
	s := newTestServer(t, false)
	ctx := context.Background()

	_, _, err := s.graphAdd(ctx, nil, GraphAddInput{Documents: []DocumentInput{
		{ID: "doc1", Content: "Maria trabalha na Ignus"},
	}})
	if err != nil {
		t.Fatalf("graphAdd: %v", err)
	}

	_, queryOut, err := s.graphQuery(ctx, nil, GraphQueryInput{Question: "onde Maria trabalha?"})
	if err != nil {
		t.Fatalf("graphQuery: %v", err)
	}
	if len(queryOut.Relations) != 1 || queryOut.Relations[0].Relation != "trabalha_em" {
		t.Fatalf("relações inesperadas: %+v", queryOut.Relations)
	}
}

func TestTreeAddThenQuery(t *testing.T) {
	s := newTestServer(t, false)
	ctx := context.Background()

	_, _, err := s.treeAdd(ctx, nil, TreeAddInput{
		ID: "doc1", Title: "Manual", Markdown: "# Cap 1\nconteúdo 1\n# Cap 2\nconteúdo 2",
	})
	if err != nil {
		t.Fatalf("treeAdd: %v", err)
	}

	_, queryOut, err := s.treeQuery(ctx, nil, TreeQueryInput{Question: "o que diz o manual?"})
	if err != nil {
		t.Fatalf("treeQuery: %v", err)
	}
	if len(queryOut.Nodes) == 0 {
		t.Fatalf("esperava ao menos 1 nó, veio vazio")
	}
}

func TestSQLQuery(t *testing.T) {
	s := newTestServer(t, true)
	ctx := context.Background()

	_, out, err := s.sqlQuery(ctx, nil, SQLQueryInput{Question: "quantos registros existem?"})
	if err != nil {
		t.Fatalf("sqlQuery: %v", err)
	}
	if out.SQL == "" {
		t.Fatalf("esperava SQL gerado")
	}
}

func TestServerMetricsRoute(t *testing.T) {
	s := newTestServer(t, false)

	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()

	s.handleMetrics(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("esperava HTTP 200, obtido %d", resp.StatusCode)
	}

	contentType := resp.Header.Get("Content-Type")
	if !strings.HasPrefix(contentType, "text/plain") {
		t.Errorf("esperava Content-Type text/plain, obtido %q", contentType)
	}
}
