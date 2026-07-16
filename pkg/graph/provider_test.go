package graph

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"urag-go/pkg/rag"
)

// TestNewOpenAIProviderExtractsViaChatCompletions prova que Config.LLMProvider
// = "openai" bate no endpoint /chat/completions (não /api/generate do Ollama)
// e usa a resposta pra extrair entidades/relações — sem depender de rede real,
// via um servidor fake que fala o formato OpenAI-compatível.
func TestNewOpenAIProviderExtractsViaChatCompletions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q, esperava /chat/completions", r.URL.Path)
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{
				"content": `{"entities":[{"name":"Maria","type":"Pessoa"},{"name":"Ignus","type":"Empresa"}],"relations":[{"source":"Maria","target":"Ignus","relation":"trabalha_em"}]}`,
			}}},
		})
	}))
	defer server.Close()

	g, err := New(Config{LLMProvider: "openai", LLMModel: "gpt-4o-mini", LLMBaseURL: server.URL, LLMAPIKey: "test-key"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if err := g.AddDocuments(context.Background(), []rag.Document{{ID: "doc1", Content: "Maria trabalha na Ignus"}}); err != nil {
		t.Fatalf("AddDocuments: %v", err)
	}

	results, err := g.Query(context.Background(), "onde Maria trabalha?", 1)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(results) != 1 || results[0].Relation != "trabalha_em" {
		t.Fatalf("results = %+v, esperava 1 relação trabalha_em", results)
	}
}

func TestNewUnknownProviderErrors(t *testing.T) {
	if _, err := New(Config{LLMProvider: "bogus", LLMModel: "x"}); err == nil {
		t.Fatal("esperava erro para provider desconhecido")
	}
}
