package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCompleteParsesChoiceContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q, esperava /chat/completions", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, esperava Bearer test-key", got)
		}

		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["model"] != "gpt-4o-mini" {
			t.Errorf("model = %v, esperava gpt-4o-mini", body["model"])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]string{"content": "resposta do modelo"}},
			},
		})
	}))
	defer server.Close()

	got, err := Complete(context.Background(), server.URL, "test-key", "gpt-4o-mini", "pergunta", false)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got != "resposta do modelo" {
		t.Errorf("got = %q, esperava %q", got, "resposta do modelo")
	}
}

func TestCompleteJSONFormatSetsResponseFormat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		rf, ok := body["response_format"].(map[string]any)
		if !ok || rf["type"] != "json_object" {
			t.Errorf("response_format = %v, esperava {type: json_object}", body["response_format"])
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{"content": "{}"}}},
		})
	}))
	defer server.Close()

	if _, err := Complete(context.Background(), server.URL, "", "model", "prompt", true); err != nil {
		t.Fatalf("Complete: %v", err)
	}
}

func TestCompleteErrorsOnNonOKStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"error":"invalid api key"}`))
	}))
	defer server.Close()

	_, err := Complete(context.Background(), server.URL, "bad-key", "model", "prompt", false)
	if err == nil || !strings.Contains(err.Error(), "401") {
		t.Fatalf("err = %v, esperava erro contendo 401", err)
	}
}
