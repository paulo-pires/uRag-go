package graph

import (
	"context"
	"strings"
	"testing"

	"urag-go/pkg/graph/persist"
	"urag-go/pkg/rag"
)

// fakeExtraction simula respostas de LLM por conteúdo do prompt, sem depender
// de Ollama rodando.
func fakeExtraction(_ context.Context, prompt string) (string, error) {
	switch {
	case strings.Contains(prompt, "Maria trabalha"):
		return `{"entities":[{"name":"Maria","type":"Pessoa"},{"name":"Ignus","type":"Empresa"}],
		"relations":[{"source":"Maria","target":"Ignus","relation":"trabalha_em"}]}`, nil
	case strings.Contains(prompt, "Ignus fica"):
		return `{"entities":[{"name":"Ignus","type":"Empresa"},{"name":"Brasil","type":"País"}],
		"relations":[{"source":"Ignus","target":"Brasil","relation":"localizada_em"}]}`, nil
	default:
		return `{}`, nil
	}
}

func TestGraphStoreAddQueryMultiHop(t *testing.T) {
	g := newGraphStoreWithCompletion(fakeExtraction, persist.Config{})

	docs := []rag.Document{
		{ID: "doc1", Content: "Maria trabalha na empresa Ignus."},
		{ID: "doc2", Content: "Ignus fica no Brasil."},
	}
	if err := g.AddDocuments(context.Background(), docs); err != nil {
		t.Fatalf("AddDocuments: %v", err)
	}

	results, err := g.Query(context.Background(), "onde Maria trabalha?", 2)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}

	var gotTrabalha, gotLocalizada bool
	for _, r := range results {
		if r.Relation == "trabalha_em" && r.Source == "maria" && r.Target == "ignus" {
			gotTrabalha = true
		}
		if r.Relation == "localizada_em" && r.Source == "ignus" && r.Target == "brasil" {
			gotLocalizada = true
		}
	}
	if !gotTrabalha {
		t.Errorf("esperava relação trabalha_em maria->ignus (1-hop), resultados: %+v", results)
	}
	if !gotLocalizada {
		t.Errorf("esperava relação localizada_em ignus->brasil (2-hop), resultados: %+v", results)
	}
}

func TestGraphStoreQueryOneHopStopsBeforeSecondHop(t *testing.T) {
	g := newGraphStoreWithCompletion(fakeExtraction, persist.Config{})

	docs := []rag.Document{
		{ID: "doc1", Content: "Maria trabalha na empresa Ignus."},
		{ID: "doc2", Content: "Ignus fica no Brasil."},
	}
	if err := g.AddDocuments(context.Background(), docs); err != nil {
		t.Fatalf("AddDocuments: %v", err)
	}

	results, err := g.Query(context.Background(), "onde Maria trabalha?", 1)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	for _, r := range results {
		if r.Relation == "localizada_em" {
			t.Errorf("hops=1 não deveria alcançar a relação de 2-hop, resultados: %+v", results)
		}
	}
}

func TestGraphStoreAddDocumentsSkipsInvalidJSON(t *testing.T) {
	calls := 0
	badThenGood := func(_ context.Context, _ string) (string, error) {
		calls++
		if calls == 1 {
			return "isso não é json", nil
		}
		return `{"entities":[{"name":"X","type":"T"}],"relations":[]}`, nil
	}

	g := newGraphStoreWithCompletion(badThenGood, persist.Config{})
	docs := []rag.Document{
		{ID: "bad", Content: "..."},
		{ID: "good", Content: "..."},
	}
	if err := g.AddDocuments(context.Background(), docs); err != nil {
		t.Fatalf("AddDocuments: %v", err)
	}
	if _, ok := g.entities["x"]; !ok {
		t.Errorf("esperava entidade 'x' do segundo doc mesmo com o primeiro retornando JSON inválido")
	}
}
