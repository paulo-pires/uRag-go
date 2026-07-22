package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	chromem "github.com/philippgille/chromem-go"
)

// qdrantStore implementa vectorBackend via Qdrant REST API (porta 6333).
// Vetores ficam no SSD (on_disk=true); só o índice HNSW fica em RAM no Qdrant.
type qdrantStore struct {
	baseURL       string
	collection    string
	apiKey        string
	embeddingFunc chromem.EmbeddingFunc
	client        *http.Client

	initOnce sync.Once
	initErr  error
}

func newQdrantStore(cfg Config, embFn chromem.EmbeddingFunc) *qdrantStore {
	base := cfg.QdrantURL
	if base == "" {
		base = "http://localhost:6333"
	}
	coll := cfg.QdrantCollection
	if coll == "" {
		coll = "documents"
	}
	return &qdrantStore{
		baseURL:       strings.TrimRight(base, "/"),
		collection:    coll,
		apiKey:        cfg.QdrantAPIKey,
		embeddingFunc: embFn,
		client:        &http.Client{Timeout: 30 * time.Second},
	}
}

// ensureCollection cria a collection se não existir (idempotente, lazy no primeiro add).
func (q *qdrantStore) ensureCollection(ctx context.Context, vecSize int) error {
	q.initOnce.Do(func() {
		body, _ := json.Marshal(map[string]any{
			"vectors": map[string]any{
				"size":     vecSize,
				"distance": "Cosine",
				"on_disk":  true, // vetores no SSD
			},
			"hnsw_config": map[string]any{
				"m":            16,
				"ef_construct": 100,
				"on_disk":      false, // índice HNSW em RAM para performance
			},
			"on_disk_payload": true, // metadata no SSD
		})
		resp, err := q.do(ctx, http.MethodPut, "/collections/"+q.collection, body)
		if err != nil {
			q.initErr = err
			return
		}
		defer resp.Body.Close()
		// 200 = criada; 409 = já existe — ambos OK
		if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusConflict {
			b, _ := io.ReadAll(resp.Body)
			q.initErr = fmt.Errorf("qdrant: criar collection: status %d: %s", resp.StatusCode, b)
		}
	})
	return q.initErr
}

func (q *qdrantStore) add(ctx context.Context, docs []Document) error {
	if len(docs) == 0 {
		return nil
	}

	points := make([]map[string]any, 0, len(docs))
	for _, d := range docs {
		vec, err := q.embeddingFunc(ctx, d.Content)
		if err != nil {
			return fmt.Errorf("qdrant: embedding doc %s: %w", d.ID, err)
		}
		if err := q.ensureCollection(ctx, len(vec)); err != nil {
			return err
		}

		payload := map[string]any{
			"content": d.Content,
			"source":  d.Source,
		}
		for k, v := range d.Meta {
			payload[k] = v
		}
		points = append(points, map[string]any{
			"id":      d.ID,
			"vector":  vec,
			"payload": payload,
		})
	}

	body, _ := json.Marshal(map[string]any{"points": points})
	resp, err := q.do(ctx, http.MethodPut, "/collections/"+q.collection+"/points?wait=true", body)
	if err != nil {
		return fmt.Errorf("qdrant: upsert: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("qdrant: upsert status %d: %s", resp.StatusCode, b)
	}
	return nil
}

func (q *qdrantStore) query(ctx context.Context, question string, topK int, where, whereDocument map[string]string) ([]SearchResult, error) {
	if topK <= 0 {
		return nil, nil
	}
	vec, err := q.embeddingFunc(ctx, question)
	if err != nil {
		return nil, fmt.Errorf("qdrant: embedding query: %w", err)
	}

	req := map[string]any{
		"query":        vec,
		"limit":        topK,
		"with_payload": true,
	}

	var must []map[string]any
	for k, v := range where {
		must = append(must, map[string]any{
			"key":   k,
			"match": map[string]any{"value": v},
		})
	}
	if v, ok := whereDocument["$contains"]; ok {
		must = append(must, map[string]any{
			"key":   "content",
			"match": map[string]any{"text": v},
		})
	}
	if len(must) > 0 {
		req["filter"] = map[string]any{"must": must}
	}

	body, _ := json.Marshal(req)
	resp, err := q.do(ctx, http.MethodPost, "/collections/"+q.collection+"/points/query", body)
	if err != nil {
		return nil, fmt.Errorf("qdrant: query: %w", err)
	}
	defer resp.Body.Close()

	var result struct {
		Result []struct {
			ID      string         `json:"id"`
			Score   float32        `json:"score"`
			Payload map[string]any `json:"payload"`
		} `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("qdrant: decode query result: %w", err)
	}

	out := make([]SearchResult, 0, len(result.Result))
	for _, r := range result.Result {
		out = append(out, SearchResult{
			Document: Document{
				ID:      r.ID,
				Content: payloadStr(r.Payload, "content"),
				Source:  payloadStr(r.Payload, "source"),
				Meta:    payloadMeta(r.Payload),
			},
			Score: r.Score,
		})
	}
	return out, nil
}

func (q *qdrantStore) getByID(ctx context.Context, id string) (Document, error) {
	resp, err := q.do(ctx, http.MethodGet, "/collections/"+q.collection+"/points/"+id, nil)
	if err != nil {
		return Document{}, fmt.Errorf("qdrant: getByID: %w", err)
	}
	defer resp.Body.Close()

	var result struct {
		Result struct {
			ID      string         `json:"id"`
			Payload map[string]any `json:"payload"`
		} `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return Document{}, fmt.Errorf("qdrant: decode point: %w", err)
	}
	return Document{
		ID:      result.Result.ID,
		Content: payloadStr(result.Result.Payload, "content"),
		Source:  payloadStr(result.Result.Payload, "source"),
		Meta:    payloadMeta(result.Result.Payload),
	}, nil
}

func (q *qdrantStore) generateEmbedding(ctx context.Context, text string) ([]float32, error) {
	return q.embeddingFunc(ctx, text)
}

func (q *qdrantStore) do(ctx context.Context, method, path string, body []byte) (*http.Response, error) {
	var r io.Reader
	if body != nil {
		r = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, q.baseURL+path, r)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if q.apiKey != "" {
		req.Header.Set("api-key", q.apiKey)
	}
	return q.client.Do(req)
}

func payloadStr(p map[string]any, key string) string {
	if v, ok := p[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

func payloadMeta(p map[string]any) map[string]string {
	out := make(map[string]string, len(p))
	for k, v := range p {
		if k == "content" || k == "source" {
			continue
		}
		if s, ok := v.(string); ok {
			out[k] = s
		} else {
			b, _ := json.Marshal(v)
			out[k] = string(b)
		}
	}
	return out
}
