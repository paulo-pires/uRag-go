package rag

import (
	"context"
	"hash/fnv"
	"math"
	"testing"
)

// fakeEmbedding gera um vetor determinístico e normalizado a partir do hash do texto,
// evitando dependência de Ollama/OpenAI rodando durante o teste.
func fakeEmbedding(_ context.Context, text string) ([]float32, error) {
	h := fnv.New32a()
	h.Write([]byte(text))
	seed := h.Sum32()

	vec := make([]float32, 8)
	var sumSq float64
	for i := range vec {
		v := float32((seed>>uint(i)&0xFF))/255 + 0.001
		vec[i] = v
		sumSq += float64(v * v)
	}
	norm := float32(math.Sqrt(sumSq))
	for i := range vec {
		vec[i] /= norm
	}
	return vec, nil
}

func TestVectorStoreAddQuery(t *testing.T) {
	vs, err := newVectorStoreWithEmbedding(Config{}, fakeEmbedding)
	if err != nil {
		t.Fatalf("newVectorStoreWithEmbedding: %v", err)
	}

	docs := []Document{
		{ID: "doc1", Content: "gatos gostam de dormir", Source: "notion"},
		{ID: "doc2", Content: "carros precisam de gasolina", Source: "pdf"},
	}
	if err := vs.add(context.Background(), docs); err != nil {
		t.Fatalf("add: %v", err)
	}

	results, err := vs.query(context.Background(), "gatos gostam de dormir", 1, nil, nil)
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("esperado 1 resultado, obtido %d", len(results))
	}
	if results[0].Document.ID != "doc1" {
		t.Errorf("esperado doc1, obtido %s", results[0].Document.ID)
	}
	if results[0].Document.Source != "notion" {
		t.Errorf("esperado source=notion, obtido %s", results[0].Document.Source)
	}
}

func TestVectorStoreQueryWhereFilter(t *testing.T) {
	vs, err := newVectorStoreWithEmbedding(Config{}, fakeEmbedding)
	if err != nil {
		t.Fatalf("newVectorStoreWithEmbedding: %v", err)
	}

	docs := []Document{
		{ID: "doc1", Content: "gatos gostam de dormir", Source: "notion"},
		{ID: "doc2", Content: "gatos gostam de dormir", Source: "pdf"},
	}
	if err := vs.add(context.Background(), docs); err != nil {
		t.Fatalf("add: %v", err)
	}

	results, err := vs.query(context.Background(), "gatos gostam de dormir", 2, map[string]string{"source": "pdf"}, nil)
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("esperado 1 resultado com where={source:pdf}, obtido %d", len(results))
	}
	if results[0].Document.ID != "doc2" {
		t.Errorf("esperado doc2, obtido %s", results[0].Document.ID)
	}
}
