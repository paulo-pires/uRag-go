// Package ollama chama o endpoint HTTP /api/generate do Ollama diretamente,
// sem SDK novo — mesmo padrão que chromem-go usa internamente para embeddings.
package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

const DefaultBaseURL = "http://localhost:11434"

// Complete pede uma resposta de texto para um prompt via Ollama /api/generate.
// jsonFormat força o Ollama a devolver JSON estrito (usado por extrações
// estruturadas, ex: pkg/graph); deixe false para respostas de texto livre
// (ex: classificação de uma palavra, usado por pkg/router).
func Complete(ctx context.Context, baseURL, model, prompt string, jsonFormat bool) (string, error) {
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}

	reqPayload := map[string]any{
		"model":  model,
		"prompt": prompt,
		"stream": false,
	}
	if jsonFormat {
		reqPayload["format"] = "json"
	}

	reqBody, err := json.Marshal(reqPayload)
	if err != nil {
		return "", fmt.Errorf("ollama: montar request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/api/generate", bytes.NewReader(reqBody))
	if err != nil {
		return "", fmt.Errorf("ollama: criar request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("ollama: chamar ollama: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("ollama: ollama respondeu %d: %s", resp.StatusCode, data)
	}

	var out struct {
		Response string `json:"response"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", fmt.Errorf("ollama: decodificar resposta ollama: %w", err)
	}
	return out.Response, nil
}

// StreamComplete pede uma resposta de texto com streaming para um prompt via Ollama /api/generate.
func StreamComplete(ctx context.Context, baseURL, model, prompt string, onChunk func(string)) error {
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}

	reqPayload := map[string]any{
		"model":  model,
		"prompt": prompt,
		"stream": true,
	}

	reqBody, err := json.Marshal(reqPayload)
	if err != nil {
		return fmt.Errorf("ollama: montar request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/api/generate", bytes.NewReader(reqBody))
	if err != nil {
		return fmt.Errorf("ollama: criar request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("ollama: chamar ollama: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("ollama: ollama respondeu %d: %s", resp.StatusCode, data)
	}

	reader := bufio.NewReader(resp.Body)
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			line, err := reader.ReadBytes('\n')
			if err != nil {
				if err == io.EOF {
					return nil
				}
				return fmt.Errorf("ollama: ler stream: %w", err)
			}

			if len(line) == 0 {
				continue
			}

			var out struct {
				Response string `json:"response"`
				Done     bool   `json:"done"`
			}
			if err := json.Unmarshal(line, &out); err != nil {
				return fmt.Errorf("ollama: decodificar chunk: %w", err)
			}

			if out.Response != "" {
				onChunk(out.Response)
			}

			if out.Done {
				return nil
			}
		}
	}
}
