package tree

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestNewOpenAIProviderNavigatesViaChatCompletions prova que Config.LLMProvider
// = "openai" bate em /chat/completions (não /api/generate do Ollama) e usa a
// resposta pra navegar a árvore — sem depender de rede real.
func TestNewOpenAIProviderNavigatesViaChatCompletions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q, esperava /chat/completions", r.URL.Path)
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{"content": "1"}}},
		})
	}))
	defer server.Close()

	tr, err := New(Config{LLMProvider: "openai", LLMModel: "gpt-4o-mini", LLMBaseURL: server.URL, LLMAPIKey: "test-key"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if err := tr.AddDocument(context.Background(), "doc1", "Manual", "# Cap 1\nconteúdo 1\n# Cap 2\nconteúdo 2"); err != nil {
		t.Fatalf("AddDocument: %v", err)
	}

	results, err := tr.Query(context.Background(), "o que diz o manual?", 1)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(results) != 1 || results[0].Title != "Cap 1" {
		t.Fatalf("results = %+v, esperava 1 nó Cap 1 (escolhido pela resposta fake '1')", results)
	}
}

func TestNewUnknownProviderErrors(t *testing.T) {
	if _, err := New(Config{LLMProvider: "bogus", LLMModel: "x"}); err == nil {
		t.Fatal("esperava erro para provider desconhecido")
	}
}
