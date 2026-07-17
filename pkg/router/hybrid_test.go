package router

import (
	"context"
	"math"
	"strings"
	"testing"

	"urag-go/pkg/graph"
	"urag-go/pkg/rag"
	"urag-go/pkg/sql"
	"urag-go/pkg/tree"
)

func TestQueryHybrid(t *testing.T) {
	// 1. Setup do Router com Mocks
	v, err := rag.NewWithEmbedding(rag.Config{}, func(ctx context.Context, text string) ([]float32, error) {
		// Mock simples de embedding estável baseado no tamanho da string
		seed := len(text)
		vec := make([]float32, 1536)
		var sumSq float64
		for i := range vec {
			val := float32((seed >> uint(i%8) & 0xFF)) / 255
			vec[i] = val
			sumSq += float64(val * val)
		}
		norm := float32(math.Sqrt(sumSq))
		for i := range vec {
			vec[i] /= norm
		}
		return vec, nil
	})
	if err != nil {
		t.Fatalf("rag.NewWithEmbedding: %v", err)
	}

	// Mock de extração do Grafo
	g := graph.NewWithCompletion(func(_ context.Context, prompt string) (string, error) {
		if strings.Contains(prompt, "Maria") {
			return `{"entities":[{"name":"Maria","type":"Pessoa"},{"name":"Ignus","type":"Empresa"}],
			"relations":[{"source":"Maria","target":"Ignus","relation":"trabalha_em"}]}`, nil
		}
		return `{}`, nil
	})

	tr := tree.NewWithNavigator(func(_ context.Context, _ string) (string, error) {
		return "1", nil
	})
	sq, err := sql.NewWithGenerator(":memory:", func(_ context.Context, _ string) (string, error) {
		return "SELECT 1", nil
	})
	if err != nil {
		t.Fatalf("sql.NewWithGenerator: %v", err)
	}

	r := NewRouterWithClassifier(v, g, tr, sq, func(_ context.Context, _ string) (string, error) {
		return "vector", nil
	})

	// 2. Adiciona documentos de teste
	docs := []rag.Document{
		{ID: "doc1", Content: "Maria trabalha na Ignus", Source: "rh"},
		{ID: "doc2", Content: "Ignus é uma empresa de tecnologia", Source: "institucional"},
	}

	if err := r.AddDocuments(context.Background(), docs); err != nil {
		t.Fatalf("AddDocuments: %v", err)
	}

	// 3. Executa busca híbrida
	results, err := r.QueryHybrid(context.Background(), "Maria trabalha na Ignus", 2, nil)
	if err != nil {
		t.Fatalf("QueryHybrid: %v", err)
	}

	// 4. Validações
	if len(results) == 0 {
		t.Fatal("esperava obter resultados na busca híbrida")
	}

	// doc1 deve estar no resultado porque contém a pergunta semântica (Vector)
	// e foi extraído no grafo (Graph)
	foundDoc1 := false
	for _, res := range results {
		if res.Document.ID == "doc1" {
			foundDoc1 = true
			break
		}
	}

	if !foundDoc1 {
		t.Error("esperava encontrar doc1 no ranking híbrido RRF")
	}
}
