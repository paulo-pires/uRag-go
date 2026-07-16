package sql

import (
	"context"
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
)

// TestNewOpenAIProviderGeneratesViaChatCompletions prova que Config.LLMProvider
// = "openai" bate em /chat/completions (não /api/generate do Ollama) e usa a
// resposta pra gerar SQL — sem depender de rede real.
func TestNewOpenAIProviderGeneratesViaChatCompletions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q, esperava /chat/completions", r.URL.Path)
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{"content": "SELECT COUNT(*) AS total FROM t"}}},
		})
	}))
	defer server.Close()

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

	store, err := New(Config{DSN: dsn, LLMProvider: "openai", LLMModel: "gpt-4o-mini", LLMBaseURL: server.URL, LLMAPIKey: "test-key"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { store.Close() })

	rows, generatedSQL, err := store.Query(context.Background(), "quantos registros existem?")
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if generatedSQL != "SELECT COUNT(*) AS total FROM t" {
		t.Errorf("generatedSQL = %q", generatedSQL)
	}
	if len(rows) != 1 {
		t.Fatalf("rows = %+v, esperava 1 linha", rows)
	}
}

func TestNewUnknownProviderErrors(t *testing.T) {
	if _, err := New(Config{DSN: ":memory:", LLMProvider: "bogus", LLMModel: "x"}); err == nil {
		t.Fatal("esperava erro para provider desconhecido")
	}
}
