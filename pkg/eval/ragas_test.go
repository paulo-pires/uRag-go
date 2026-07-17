package eval

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestEvaluatorRAGASMetrics(t *testing.T) {
	// 1. Cria servidor HTTP Mock para responder prompts de RAGAS
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Prompt string `json:"prompt"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		var responseJSON any

		if strings.Contains(req.Prompt, "extraia todas as afirmações") {
			responseJSON = map[string]string{
				"response": `{"statements": ["Maria trabalha na Ignus", "Maria mora em Paris"]}`,
			}
		} else if strings.Contains(req.Prompt, "corroborada pelo contexto") {
			// Afirmação sobre a Ignus é verdadeira no contexto, sobre Paris é falsa
			ans := "não"
			if strings.Contains(req.Prompt, "Maria trabalha na Ignus") {
				ans = "sim"
			}
			responseJSON = map[string]string{
				"response": ans,
			}
		} else if strings.Contains(req.Prompt, "Gere exatamente 3 perguntas hipotéticas") {
			responseJSON = map[string]string{
				"response": `{"questions": ["onde Maria trabalha?", "quem é Maria?"]}`,
			}
		} else if strings.Contains(req.Prompt, "sentença da resposta ideal") {
			// Sentença do ground truth corroborada
			responseJSON = map[string]string{
				"response": "sim",
			}
		} else {
			responseJSON = map[string]string{
				"response": "sim",
			}
		}

		json.NewEncoder(w).Encode(responseJSON)
	}))
	defer server.Close()

	// 2. Setup Evaluator
	evaluator := NewEvaluator("ollama", "granite4:micro-h", server.URL, "")

	// 3. Mock do embedding generator
	embedFunc := func(ctx context.Context, text string) ([]float32, error) {
		// Retorna embeddings idênticos para simular similaridade 1.0 (perfeita)
		return []float32{1.0, 0.0, 0.0}, nil
	}

	// 4. Executa avaliação
	question := "Onde Maria trabalha?"
	answer := "Maria trabalha na Ignus e ela mora em Paris."
	contextText := "Maria é desenvolvedora de software e trabalha na Ignus desde 2024."
	groundTruth := "Maria trabalha na empresa Ignus."

	metrics, err := evaluator.Evaluate(context.Background(), question, answer, contextText, groundTruth, embedFunc)
	if err != nil {
		t.Fatalf("falha ao avaliar RAGAS: %v", err)
	}

	// 5. Validações
	// Faithfulness: 1 afirmação suportada (Ignus), 1 não suportada (Paris) -> score = 0.5
	if metrics.Faithfulness != 0.5 {
		t.Errorf("esperava Faithfulness 0.5, obtido: %.2f", metrics.Faithfulness)
	}

	// AnswerRelevancy: similaridade de cosseno com embeddings idênticos -> score = 1.0
	if metrics.AnswerRelevancy != 1.0 {
		t.Errorf("esperava AnswerRelevancy 1.0, obtido: %.2f", metrics.AnswerRelevancy)
	}

	// ContextRecall: sentença do ground truth encontrada no contexto -> score = 1.0
	if metrics.ContextRecall != 1.0 {
		t.Errorf("esperava ContextRecall 1.0, obtido: %.2f", metrics.ContextRecall)
	}
}
