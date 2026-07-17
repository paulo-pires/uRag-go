package rerank

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestReRankerLLMResponse(t *testing.T) {
	// 1. Cria servidor HTTP Mock para simular o Ollama
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Decodifica o request payload
		var req struct {
			Prompt string `json:"prompt"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		var scoreStr string
		if strings.Contains(req.Prompt, "altamente") {
			scoreStr = "9"
		} else if strings.Contains(req.Prompt, "médio") {
			scoreStr = "5"
		} else {
			scoreStr = "1"
		}

		responseJSON := map[string]string{
			"response": scoreStr,
		}
		json.NewEncoder(w).Encode(responseJSON)
	}))
	defer server.Close()

	// 2. Inicia o ReRanker apontando para o servidor mock
	reRanker := New("ollama", "granite4:micro-h", server.URL, "")

	// 3. Documentos de teste
	docs := []Result{
		{DocumentID: "doc1", Content: "Este trecho é médio", Score: 0.8},
		{DocumentID: "doc2", Content: "Este trecho é totalmente irrelevante", Score: 0.9},
		{DocumentID: "doc3", Content: "Este trecho é altamente relevante", Score: 0.2},
	}

	// 4. Executa re-ranking
	reranked, err := reRanker.ReRank(context.Background(), "qual trecho é relevante?", docs)
	if err != nil {
		t.Fatalf("falha ao executar ReRank: %v", err)
	}

	// 5. Validações: doc3 deve subir para o topo (score 0.9), doc1 em segundo (score 0.5), doc2 em último (score 0.1)
	if len(reranked) != 3 {
		t.Fatalf("esperava 3 itens re-ordenados, obtido %d", len(reranked))
	}

	if reranked[0].DocumentID != "doc3" {
		t.Errorf("esperava doc3 em primeiro lugar no re-ranking, obtido: %s (score %.2f)", reranked[0].DocumentID, reranked[0].Score)
	}

	if reranked[1].DocumentID != "doc1" {
		t.Errorf("esperava doc1 em segundo lugar, obtido: %s (score %.2f)", reranked[1].DocumentID, reranked[1].Score)
	}

	if reranked[2].DocumentID != "doc2" {
		t.Errorf("esperava doc2 em último lugar, obtido: %s (score %.2f)", reranked[2].DocumentID, reranked[2].Score)
	}
}
