// Package openai chama o endpoint HTTP /chat/completions diretamente, sem SDK
// novo — mesmo padrão de internal/ollama. baseURL é configurável de propósito:
// cobre não só a OpenAI oficial, mas qualquer provider que implemente o mesmo
// formato de API (vLLM, LM Studio, Together, Groq, etc — "OpenAI-compatível").
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

const DefaultBaseURL = "https://api.openai.com/v1"

// Complete pede uma resposta de texto para um prompt via Chat Completions API.
// jsonFormat pede response_format=json_object (suportado pela OpenAI e pela
// maioria dos providers compatíveis modernos); deixe false para texto livre.
func Complete(ctx context.Context, baseURL, apiKey, model, prompt string, jsonFormat bool) (string, error) {
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}

	reqPayload := map[string]any{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}
	if jsonFormat {
		reqPayload["response_format"] = map[string]string{"type": "json_object"}
	}

	reqBody, err := json.Marshal(reqPayload)
	if err != nil {
		return "", fmt.Errorf("openai: montar request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/chat/completions", bytes.NewReader(reqBody))
	if err != nil {
		return "", fmt.Errorf("openai: criar request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("openai: chamar provider: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("openai: provider respondeu %d: %s", resp.StatusCode, data)
	}

	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", fmt.Errorf("openai: decodificar resposta: %w", err)
	}
	if len(out.Choices) == 0 {
		return "", fmt.Errorf("openai: resposta sem choices")
	}
	return out.Choices[0].Message.Content, nil
}
