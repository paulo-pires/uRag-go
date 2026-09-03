package mcpserver

import (
	"context"
	"database/sql"
	"encoding/json"
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
	"urag-go/pkg/rollout"
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

	fakeReplay := rollout.NewReplayEngine(
		func(_ context.Context, _, _, _, _, prompt string) (string, error) {
			return "Mocked candidate response for: " + prompt, nil
		},
		func(_ context.Context, _, _, _ string) (float64, float64, error) {
			return 0.94, 0.92, nil
		},
	)

	s := &Server{
		mcp:     mcp.NewServer(&mcp.Implementation{Name: "test"}, nil),
		vector:  vector,
		graph:   graph.NewWithCompletion(fakeGraphExtraction),
		tree:    tree.NewWithNavigator(fakeTreeNavigate),
		rollout: rollout.NewManager(),
		replay:  fakeReplay,
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

func TestMCPConfigurePromptRollout(t *testing.T) {
	s := newTestServer(t, false)
	ctx := context.Background()

	inputJSON := `{
		"prompt_id": "prompt-rag-v2",
		"strategy": "canary",
		"base_prompt": "Você é um assistente RAG v1.",
		"candidate_prompt": "Você é um assistente RAG v2 otimizado.",
		"traffic_percentage": 25.0,
		"auto_rollback": true,
		"min_quality_score": 0.80,
		"max_error_rate": 0.05
	}`

	var input rollout.RolloutConfig
	if err := json.Unmarshal([]byte(inputJSON), &input); err != nil {
		t.Fatalf("falha ao deserializar input: %v", err)
	}

	_, out, err := s.configurePromptRollout(ctx, nil, input)
	if err != nil {
		t.Fatalf("configurePromptRollout: %v", err)
	}

	if out.Status != "active" {
		t.Fatalf("status esperado 'active', obtido %q", out.Status)
	}
	if out.State.PromptID != "prompt-rag-v2" {
		t.Fatalf("prompt_id incorreto no estado: %s", out.State.PromptID)
	}

	outBytes, err := json.Marshal(out)
	if err != nil {
		t.Fatalf("falha ao serializar output: %v", err)
	}
	if !strings.Contains(string(outBytes), "prompt-rag-v2") {
		t.Fatalf("output JSON esperado conter prompt-rag-v2, obtido: %s", string(outBytes))
	}
}

func TestMCPGetRolloutStatus(t *testing.T) {
	s := newTestServer(t, false)
	ctx := context.Background()

	_, _, _ = s.configurePromptRollout(ctx, nil, rollout.RolloutConfig{
		PromptID:        "p-status-test",
		Strategy:        "shadow",
		BasePrompt:      "Base",
		CandidatePrompt: "Candidate",
	})

	inputJSON := `{"prompt_id": "p-status-test"}`
	var input GetRolloutStatusInput
	if err := json.Unmarshal([]byte(inputJSON), &input); err != nil {
		t.Fatalf("falha ao deserializar input: %v", err)
	}

	_, state, err := s.getRolloutStatus(ctx, nil, input)
	if err != nil {
		t.Fatalf("getRolloutStatus: %v", err)
	}

	if state.PromptID != "p-status-test" || state.Config.Strategy != "shadow" {
		t.Fatalf("estado retornado inesperado: %+v", state)
	}

	outBytes, err := json.Marshal(state)
	if err != nil {
		t.Fatalf("falha ao serializar output: %v", err)
	}
	if !strings.Contains(string(outBytes), "p-status-test") {
		t.Fatalf("JSON não contém id: %s", string(outBytes))
	}
}

func TestMCPEvaluateRolloutDecision(t *testing.T) {
	s := newTestServer(t, false)
	ctx := context.Background()

	_, _, _ = s.configurePromptRollout(ctx, nil, rollout.RolloutConfig{
		PromptID:          "p-eval-test",
		Strategy:          "canary",
		BasePrompt:        "Prompt Base",
		CandidatePrompt:   "Prompt Candidate",
		TrafficPercentage: 50.0,
	})

	inputJSON := `{"prompt_id": "p-eval-test", "sticky_key": "tenant-42"}`
	var input EvaluateRolloutDecisionInput
	if err := json.Unmarshal([]byte(inputJSON), &input); err != nil {
		t.Fatalf("falha ao deserializar input: %v", err)
	}

	_, decision, err := s.evaluateRolloutDecision(ctx, nil, input)
	if err != nil {
		t.Fatalf("evaluateRolloutDecision: %v", err)
	}

	if decision.PromptID != "p-eval-test" {
		t.Fatalf("prompt_id incorreto na decisão: %s", decision.PromptID)
	}
	if decision.SelectedVariant != "base" && decision.SelectedVariant != "candidate" {
		t.Fatalf("variante inválida: %s", decision.SelectedVariant)
	}
	if decision.Prompt == "" {
		t.Fatal("prompt vazio na decisão")
	}

	outBytes, err := json.Marshal(decision)
	if err != nil {
		t.Fatalf("falha ao serializar output: %v", err)
	}
	if !strings.Contains(string(outBytes), "selected_variant") {
		t.Fatalf("JSON não contém selected_variant: %s", string(outBytes))
	}
}

func TestMCPReplayTraces(t *testing.T) {
	s := newTestServer(t, false)
	ctx := context.Background()

	inputJSON := `{
		"traces": [
			{
				"trace_id": "trace-101",
				"prompt_id": "p-rag",
				"input": "Como emitir nota fiscal?",
				"original_prompt": "Base prompt",
				"original_output": "Acesse o módulo fiscal e clique em emitir.",
				"original_score": 0.85,
				"original_latency_ms": 300.0,
				"original_tokens": 25
			}
		],
		"config": {
			"candidate_prompt": "Você é especialista fiscal. Pergunta: {{input}}",
			"min_fidelity": 0.80,
			"concurrency": 2
		}
	}`

	var input ReplayTracesInput
	if err := json.Unmarshal([]byte(inputJSON), &input); err != nil {
		t.Fatalf("falha ao deserializar input: %v", err)
	}

	_, summary, err := s.replayTraces(ctx, nil, input)
	if err != nil {
		t.Fatalf("replayTraces: %v", err)
	}

	if summary.TotalTraces != 1 {
		t.Fatalf("TotalTraces = %d, esperado 1", summary.TotalTraces)
	}
	if summary.SuccessfulReplays != 1 {
		t.Fatalf("SuccessfulReplays = %d, esperado 1", summary.SuccessfulReplays)
	}
	if summary.AvgFidelity <= 0 {
		t.Fatalf("AvgFidelity = %f, esperado > 0", summary.AvgFidelity)
	}
	if len(summary.Results) != 1 || !summary.Results[0].Passed {
		t.Fatalf("resultado inesperado: %+v", summary.Results)
	}

	outBytes, err := json.Marshal(summary)
	if err != nil {
		t.Fatalf("falha ao serializar output: %v", err)
	}
	if !strings.Contains(string(outBytes), "avg_fidelity") {
		t.Fatalf("JSON não contém avg_fidelity: %s", string(outBytes))
	}
}
