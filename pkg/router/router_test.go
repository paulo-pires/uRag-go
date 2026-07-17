package router

import (
	"context"
	"encoding/json"
	"hash/fnv"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"urag-go/pkg/graph"
	"urag-go/pkg/rag"
	"urag-go/pkg/rerank"
	"urag-go/pkg/sql"
	"urag-go/pkg/tree"
)

// fakeEmbedding e fakeExtraction seguem o mesmo padrão determinístico usado em
// pkg/rag/vector_test.go e pkg/graph/graph_test.go, sem depender de Ollama.
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

func fakeExtraction(_ context.Context, prompt string) (string, error) {
	if strings.Contains(prompt, "Maria trabalha") {
		return `{"entities":[{"name":"Maria","type":"Pessoa"},{"name":"Ignus","type":"Empresa"}],
		"relations":[{"source":"Maria","target":"Ignus","relation":"trabalha_em"}]}`, nil
	}
	return `{}`, nil
}

// fakeTreeNavigate sempre escolhe o primeiro item da lista — suficiente para
// os testes do Router, que só verificam se a dispatch chegou na store certa,
// não a qualidade da navegação (isso já é coberto em pkg/tree).
func fakeTreeNavigate(_ context.Context, _ string) (string, error) {
	return "1", nil
}

// fakeSQLGenerate devolve uma query fixa que não depende de nenhuma tabela
// existir — suficiente para provar que o Router despachou pro Store certo.
func fakeSQLGenerate(_ context.Context, _ string) (string, error) {
	return "SELECT 1 AS x", nil
}

func newTestRouter(t *testing.T, classify ClassifyFunc) *Router {
	t.Helper()
	v, err := rag.NewWithEmbedding(rag.Config{}, fakeEmbedding)
	if err != nil {
		t.Fatalf("rag.NewWithEmbedding: %v", err)
	}
	g := graph.NewWithCompletion(fakeExtraction)
	tr := tree.NewWithNavigator(fakeTreeNavigate)
	if err := tr.AddDocument(context.Background(), "doc1", "Doc", "# Capítulo\nConteúdo do capítulo.\n"); err != nil {
		t.Fatalf("tr.AddDocument: %v", err)
	}
	sq, err := sql.NewWithGenerator(":memory:", fakeSQLGenerate)
	if err != nil {
		t.Fatalf("sql.NewWithGenerator: %v", err)
	}
	return NewRouterWithClassifier(v, g, tr, sq, classify)
}

func TestRouterQueryDispatchesToVector(t *testing.T) {
	r := newTestRouter(t, func(_ context.Context, _ string) (string, error) { return "vector", nil })

	docs := []rag.Document{{ID: "doc1", Content: "gatos gostam de dormir", Source: "notion"}}
	if err := r.AddDocuments(context.Background(), docs); err != nil {
		t.Fatalf("AddDocuments: %v", err)
	}

	result, err := r.Query(context.Background(), "gatos gostam de dormir", 1)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if result.Strategy != StrategyVector {
		t.Fatalf("esperava StrategyVector, obtido %s", result.Strategy)
	}
	if len(result.Vector) != 1 || result.Vector[0].Document.ID != "doc1" {
		t.Errorf("resultado vector inesperado: %+v", result.Vector)
	}
	if result.Graph != nil {
		t.Errorf("esperava Graph vazio quando Strategy=vector, obtido %+v", result.Graph)
	}
}

func TestRouterQueryDispatchesToGraph(t *testing.T) {
	r := newTestRouter(t, func(_ context.Context, _ string) (string, error) { return "graph", nil })

	docs := []rag.Document{{ID: "doc1", Content: "Maria trabalha na empresa Ignus."}}
	if err := r.AddDocuments(context.Background(), docs); err != nil {
		t.Fatalf("AddDocuments: %v", err)
	}

	result, err := r.Query(context.Background(), "onde Maria trabalha?", 5)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if result.Strategy != StrategyGraph {
		t.Fatalf("esperava StrategyGraph, obtido %s", result.Strategy)
	}
	if len(result.Graph) != 1 || result.Graph[0].Relation != "trabalha_em" {
		t.Errorf("resultado graph inesperado: %+v", result.Graph)
	}
	if result.Vector != nil {
		t.Errorf("esperava Vector vazio quando Strategy=graph, obtido %+v", result.Vector)
	}
}

func TestRouterQueryDispatchesToBoth(t *testing.T) {
	r := newTestRouter(t, func(_ context.Context, _ string) (string, error) { return "both", nil })

	docs := []rag.Document{
		{ID: "doc1", Content: "gatos gostam de dormir", Source: "notion"},
		{ID: "doc2", Content: "Maria trabalha na empresa Ignus."},
	}
	if err := r.AddDocuments(context.Background(), docs); err != nil {
		t.Fatalf("AddDocuments: %v", err)
	}

	result, err := r.Query(context.Background(), "gatos e Maria", 2)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if result.Strategy != StrategyBoth {
		t.Fatalf("esperava StrategyBoth, obtido %s", result.Strategy)
	}
	if len(result.Vector) == 0 {
		t.Errorf("esperava Vector preenchido quando Strategy=both, obtido vazio")
	}
	if len(result.Graph) == 0 {
		t.Errorf("esperava Graph preenchido quando Strategy=both, obtido vazio")
	}
}

func TestRouterQueryDispatchesToTree(t *testing.T) {
	r := newTestRouter(t, func(_ context.Context, _ string) (string, error) { return "tree", nil })

	result, err := r.Query(context.Background(), "o que diz o capítulo?", 3)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if result.Strategy != StrategyTree {
		t.Fatalf("esperava StrategyTree, obtido %s", result.Strategy)
	}
	if len(result.Tree) == 0 {
		t.Errorf("esperava Tree preenchido quando Strategy=tree, obtido vazio")
	}
	if result.Vector != nil || result.Graph != nil {
		t.Errorf("esperava Vector/Graph vazios quando Strategy=tree, obtido vector=%+v graph=%+v", result.Vector, result.Graph)
	}
}

func TestRouterQueryDispatchesToSQL(t *testing.T) {
	r := newTestRouter(t, func(_ context.Context, _ string) (string, error) { return "sql", nil })

	result, err := r.Query(context.Background(), "quantos registros existem?", 3)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if result.Strategy != StrategySQL {
		t.Fatalf("esperava StrategySQL, obtido %s", result.Strategy)
	}
	if result.SQLQuery != "SELECT 1 AS x" {
		t.Errorf("SQLQuery inesperado: %q", result.SQLQuery)
	}
	if len(result.SQLRows) != 1 {
		t.Errorf("esperava 1 linha de resultado, obtido %d: %+v", len(result.SQLRows), result.SQLRows)
	}
}

func TestRouterQueryFallsBackToVectorOnAmbiguousClassification(t *testing.T) {
	r := newTestRouter(t, func(_ context.Context, _ string) (string, error) { return "sei lá, talvez", nil })

	docs := []rag.Document{{ID: "doc1", Content: "gatos gostam de dormir", Source: "notion"}}
	if err := r.AddDocuments(context.Background(), docs); err != nil {
		t.Fatalf("AddDocuments: %v", err)
	}

	result, err := r.Query(context.Background(), "qualquer pergunta", 1)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if result.Strategy != StrategyVector {
		t.Fatalf("esperava fallback StrategyVector, obtido %s", result.Strategy)
	}
}

func TestRouterQueryFusedReturnsBoth(t *testing.T) {
	// classify não é chamado por QueryFused — passar um fake que falha se for
	// invocado garante que o teste realmente exercita o caminho sem classificação.
	r := newTestRouter(t, func(_ context.Context, _ string) (string, error) {
		t.Fatal("classify não deveria ser chamado por QueryFused")
		return "", nil
	})

	docs := []rag.Document{
		{ID: "doc1", Content: "gatos gostam de dormir", Source: "notion"},
		{ID: "doc2", Content: "Maria trabalha na empresa Ignus."},
	}
	if err := r.AddDocuments(context.Background(), docs); err != nil {
		t.Fatalf("AddDocuments: %v", err)
	}

	result, err := r.QueryFused(context.Background(), "gatos e Maria", 2)
	if err != nil {
		t.Fatalf("QueryFused: %v", err)
	}
	if len(result.Vector) == 0 {
		t.Errorf("esperava resultados vector preenchidos, obtido vazio")
	}
	if len(result.Graph) == 0 {
		t.Errorf("esperava resultados graph preenchidos, obtido vazio")
	}
}

func TestRouterQueryFallsBackToVectorOnClassifyError(t *testing.T) {
	r := newTestRouter(t, func(_ context.Context, _ string) (string, error) {
		return "", context.DeadlineExceeded
	})

	docs := []rag.Document{{ID: "doc1", Content: "gatos gostam de dormir", Source: "notion"}}
	if err := r.AddDocuments(context.Background(), docs); err != nil {
		t.Fatalf("AddDocuments: %v", err)
	}

	result, err := r.Query(context.Background(), "qualquer pergunta", 1)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if result.Strategy != StrategyVector {
		t.Fatalf("esperava fallback StrategyVector quando classify falha, obtido %s", result.Strategy)
	}
}

func TestRouterQueryWithReRanker(t *testing.T) {
	r := newTestRouter(t, func(_ context.Context, _ string) (string, error) { return "vector", nil })

	// Cria servidor HTTP Mock para simular o re-ranker
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Prompt string `json:"prompt"`
		}
		json.NewDecoder(r.Body).Decode(&req)

		w.Header().Set("Content-Type", "application/json")
		scoreStr := "2"
		if strings.Contains(req.Prompt, "dormir") {
			scoreStr = "10"
		}
		responseJSON := map[string]string{
			"response": scoreStr,
		}
		json.NewEncoder(w).Encode(responseJSON)
	}))
	defer server.Close()

	r.reranker = rerank.New("ollama", "granite4:micro-h", server.URL, "")

	docs := []rag.Document{
		{ID: "doc1", Content: "cachorros gostam de correr", Source: "notion"},
		{ID: "doc2", Content: "gatos gostam de dormir", Source: "notion"},
	}
	if err := r.AddDocuments(context.Background(), docs); err != nil {
		t.Fatalf("AddDocuments: %v", err)
	}

	// Sem re-ranking, cachorros poderia vir antes por similaridade padrão do mock.
	// Com o re-ranking, o mock dá nota 10 para o prompt contendo "gatos" e 2 para "cachorros".
	// Então o doc2 (gatos) DEVE ser re-ordenado para primeiro lugar!
	result, err := r.Query(context.Background(), "qualquer coisa sobre gatos", 2)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}

	if len(result.Vector) < 2 {
		t.Fatalf("esperava 2 resultados, obtido %d", len(result.Vector))
	}

	if result.Vector[0].Document.ID != "doc2" {
		t.Errorf("esperava doc2 (gatos) em primeiro após re-ranking, obtido: %s", result.Vector[0].Document.ID)
	}
}
