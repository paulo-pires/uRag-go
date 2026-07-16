package rag

import (
	"context"
	"testing"
)

func TestVectorStoreANNQuery(t *testing.T) {
	vs, err := newVectorStoreWithEmbedding(Config{Index: "hnsw"}, fakeEmbedding)
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
}

func TestVectorStoreANNWhereFilter(t *testing.T) {
	vs, err := newVectorStoreWithEmbedding(Config{Index: "hnsw"}, fakeEmbedding)
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
	if len(results) != 1 || results[0].Document.ID != "doc2" {
		t.Fatalf("esperado 1 resultado (doc2, source=pdf), obtido %+v", results)
	}
}

func TestVectorStoreANNRejectsWhereDocument(t *testing.T) {
	vs, err := newVectorStoreWithEmbedding(Config{Index: "hnsw"}, fakeEmbedding)
	if err != nil {
		t.Fatalf("newVectorStoreWithEmbedding: %v", err)
	}
	_, err = vs.query(context.Background(), "x", 1, nil, map[string]string{"$contains": "y"})
	if err == nil {
		t.Fatal("esperava erro ao usar whereDocument com Index=hnsw")
	}
}

func TestNewVectorStoreRejectsHNSWWithPersistPath(t *testing.T) {
	_, err := newVectorStoreWithEmbedding(Config{Index: "hnsw", PersistPath: "should-not-be-created.db"}, fakeEmbedding)
	if err == nil {
		t.Fatal("esperava erro ao combinar Index=hnsw com PersistPath")
	}
}

func TestNewVectorStoreRejectsUnknownIndex(t *testing.T) {
	_, err := newVectorStoreWithEmbedding(Config{Index: "ivfflat"}, fakeEmbedding)
	if err == nil {
		t.Fatal("esperava erro para Index desconhecido")
	}
}
