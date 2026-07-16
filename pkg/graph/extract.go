package graph

import (
	"context"

	"urag-go/internal/ollama"
	"urag-go/internal/openai"
)

// completionFunc pede uma resposta em JSON para um prompt. Implementação real
// chama Ollama /api/generate ou um provider OpenAI-compatível via
// /chat/completions; testes injetam uma fake, mesmo padrão do fakeEmbedding
// em pkg/rag/vector_test.go.
type completionFunc func(ctx context.Context, prompt string) (string, error)

// ollamaCompletion pede saída em JSON estrito via internal/ollama.Complete.
func ollamaCompletion(model, baseURL string) completionFunc {
	return func(ctx context.Context, prompt string) (string, error) {
		return ollama.Complete(ctx, baseURL, model, prompt, true)
	}
}

// openaiCompletion pede saída em JSON estrito via internal/openai.Complete —
// serve tanto a OpenAI oficial quanto qualquer provider que implemente o
// mesmo formato de Chat Completions (baseURL configurável).
func openaiCompletion(model, baseURL, apiKey string) completionFunc {
	return func(ctx context.Context, prompt string) (string, error) {
		return openai.Complete(ctx, baseURL, apiKey, model, prompt, true)
	}
}
