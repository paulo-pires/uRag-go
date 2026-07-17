package eval

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"urag-go/internal/ollama"
	"urag-go/internal/openai"
)

type Metrics struct {
	Faithfulness    float64 // Fidelidade: resposta apoiada no contexto
	AnswerRelevancy float64 // Relevância da resposta para a pergunta
	ContextRecall   float64 // Recall: contexto cobre a resposta ideal (ground truth)
}

type Evaluator struct {
	provider string
	model    string
	baseURL  string
	apiKey   string
}

// NewEvaluator cria um avaliador de métricas RAGAS.
func NewEvaluator(provider, model, baseURL, apiKey string) *Evaluator {
	if model == "" {
		model = "granite4:micro-h"
	}
	return &Evaluator{
		provider: provider,
		model:    model,
		baseURL:  baseURL,
		apiKey:   apiKey,
	}
}

// Evaluate calcula as métricas RAGAS sobre o RAG.
func (e *Evaluator) Evaluate(ctx context.Context, question, answer, contextText, groundTruth string, embedFunc func(context.Context, string) ([]float32, error)) (Metrics, error) {
	var metrics Metrics

	// 1. Calcula Faithfulness (Fidelidade)
	faithfulness, err := e.evalFaithfulness(ctx, answer, contextText)
	if err != nil {
		return metrics, fmt.Errorf("faithfulness: %w", err)
	}
	metrics.Faithfulness = faithfulness

	// 2. Calcula Answer Relevancy
	if embedFunc != nil {
		relevancy, err := e.evalAnswerRelevancy(ctx, question, answer, embedFunc)
		if err != nil {
			return metrics, fmt.Errorf("answer relevancy: %w", err)
		}
		metrics.AnswerRelevancy = relevancy
	}

	// 3. Calcula Context Recall (se groundTruth estiver presente)
	if groundTruth != "" {
		recall, err := e.evalContextRecall(ctx, groundTruth, contextText)
		if err != nil {
			return metrics, fmt.Errorf("context recall: %w", err)
		}
		metrics.ContextRecall = recall
	}

	return metrics, nil
}

func (e *Evaluator) callLLM(ctx context.Context, prompt string, jsonFormat bool) (string, error) {
	if e.provider == "openai" {
		return openai.Complete(ctx, e.baseURL, e.apiKey, e.model, prompt, jsonFormat)
	}
	return ollama.Complete(ctx, e.baseURL, e.model, prompt, jsonFormat)
}

func (e *Evaluator) evalFaithfulness(ctx context.Context, answer, contextText string) (float64, error) {
	if strings.TrimSpace(answer) == "" || strings.TrimSpace(contextText) == "" {
		return 0.0, nil
	}

	// Extrai afirmações (statements) em formato JSON
	prompt := fmt.Sprintf(`Dada a resposta abaixo, extraia todas as afirmações fatuais contidas nela.
Retorne o resultado estritamente no formato JSON abaixo, sem qualquer outro texto:
{
  "statements": [
    "afirmação 1",
    "afirmação 2"
  ]
}

Resposta: %s`, answer)

	resp, err := e.callLLM(ctx, prompt, true)
	if err != nil {
		return 0.0, err
	}

	var out struct {
		Statements []string `json:"statements"`
	}
	if err := json.Unmarshal([]byte(resp), &out); err != nil {
		// Tolerância para formato LLM que enrola JSON
		cleaned := cleanJSONString(resp)
		if err := json.Unmarshal([]byte(cleaned), &out); err != nil {
			return 0.0, fmt.Errorf("extrair afirmações: %w", err)
		}
	}

	if len(out.Statements) == 0 {
		return 1.0, nil // nada a contestar
	}

	supportedCount := 0
	for _, statement := range out.Statements {
		checkPrompt := fmt.Sprintf(`Verifique se a afirmação abaixo é sustentada e corroborada pelo contexto fornecido.
Afirmação: %s
Contexto: %s

Responda apenas "sim" se for sustentada, ou "não" se não for sustentada. Não dê explicações.
Resposta:`, statement, contextText)

		ans, err := e.callLLM(ctx, checkPrompt, false)
		if err != nil {
			continue
		}
		if strings.Contains(strings.ToLower(ans), "sim") {
			supportedCount++
		}
	}

	return float64(supportedCount) / float64(len(out.Statements)), nil
}

func (e *Evaluator) evalAnswerRelevancy(ctx context.Context, question, answer string, embedFunc func(context.Context, string) ([]float32, error)) (float64, error) {
	if strings.TrimSpace(question) == "" || strings.TrimSpace(answer) == "" {
		return 0.0, nil
	}

	// Gera 3 perguntas hipotéticas baseadas na resposta
	prompt := fmt.Sprintf(`Gere exatamente 3 perguntas hipotéticas diferentes que a resposta abaixo responde diretamente.
Retorne o resultado estritamente no formato JSON abaixo, sem qualquer outro texto:
{
  "questions": [
    "pergunta 1",
    "pergunta 2",
    "pergunta 3"
  ]
}

Resposta: %s`, answer)

	resp, err := e.callLLM(ctx, prompt, true)
	if err != nil {
		return 0.0, err
	}

	var out struct {
		Questions []string `json:"questions"`
	}
	if err := json.Unmarshal([]byte(resp), &out); err != nil {
		cleaned := cleanJSONString(resp)
		if err := json.Unmarshal([]byte(cleaned), &out); err != nil {
			return 0.0, fmt.Errorf("gerar perguntas hipotéticas: %w", err)
		}
	}

	if len(out.Questions) == 0 {
		return 0.0, nil
	}

	// Obtém embedding da pergunta original
	qEmbed, err := embedFunc(ctx, question)
	if err != nil {
		return 0.0, err
	}

	var sumSimilarity float64
	validCount := 0

	for _, hypQ := range out.Questions {
		hypEmbed, err := embedFunc(ctx, hypQ)
		if err != nil {
			continue
		}
		sumSimilarity += cosineSimilarity(qEmbed, hypEmbed)
		validCount++
	}

	if validCount == 0 {
		return 0.0, nil
	}

	return sumSimilarity / float64(validCount), nil
}

func (e *Evaluator) evalContextRecall(ctx context.Context, groundTruth, contextText string) (float64, error) {
	if strings.TrimSpace(groundTruth) == "" || strings.TrimSpace(contextText) == "" {
		return 0.0, nil
	}

	// Quebra o ground truth em sentenças simples
	rawSentences := strings.Split(groundTruth, ".")
	var sentences []string
	for _, s := range rawSentences {
		trimmed := strings.TrimSpace(s)
		if len(trimmed) > 5 { // ignora pedaços vazios ou curtos demais
			sentences = append(sentences, trimmed)
		}
	}

	if len(sentences) == 0 {
		return 0.0, nil
	}

	recalledCount := 0
	for _, sentence := range sentences {
		checkPrompt := fmt.Sprintf(`Verifique se a informação contida na sentença da resposta ideal pode ser encontrada ou inferida a partir do contexto fornecido.
Sentença: %s
Contexto: %s

Responda apenas "sim" se puder ser inferida, ou "não" se não puder. Não dê explicações.
Resposta:`, sentence, contextText)

		ans, err := e.callLLM(ctx, checkPrompt, false)
		if err != nil {
			continue
		}
		if strings.Contains(strings.ToLower(ans), "sim") {
			recalledCount++
		}
	}

	return float64(recalledCount) / float64(len(sentences)), nil
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0
	}
	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i] * b[i])
		normA += float64(a[i] * a[i])
		normB += float64(b[i] * b[i])
	}
	if normA == 0.0 || normB == 0.0 {
		return 0.0
	}
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

func cleanJSONString(s string) string {
	// Remove blocos de markdown ```json ... ``` se o LLM os adicionou
	if idx := strings.Index(s, "```json"); idx != -1 {
		s = s[idx+7:]
	} else if idx := strings.Index(s, "```"); idx != -1 {
		s = s[idx+3:]
	}
	if idx := strings.LastIndex(s, "```"); idx != -1 {
		s = s[:idx]
	}
	return strings.TrimSpace(s)
}