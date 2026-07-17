// Package router provê o roteamento e a busca híbrida.
package router

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"urag-go/pkg/graph"
	"urag-go/pkg/rag"
)

// QueryHybrid executa buscas em paralelo no Vector RAG e Graph RAG,
// e funde os rankings usando RRF (Reciprocal Rank Fusion).
func (r *Router) QueryHybrid(ctx context.Context, question string, topK int, weights map[string]float64) ([]rag.SearchResult, error) {
	if weights == nil {
		weights = map[string]float64{"vector": 1.0, "graph": 1.0}
	}

	var (
		vecResults   []rag.SearchResult
		graphResults []graph.Relation
		vecErr       error
		graphErr     error
	)

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		vecResults, vecErr = r.vector.Query(ctx, question, topK)
	}()

	go func() {
		defer wg.Done()
		graphResults, graphErr = r.graph.Query(ctx, question, graphQueryHops)
	}()

	wg.Wait()

	if vecErr != nil {
		return nil, fmt.Errorf("hybrid query: vector query: %w", vecErr)
	}
	if graphErr != nil {
		return nil, fmt.Errorf("hybrid query: graph query: %w", graphErr)
	}

	// 1. Extrai DocIDs da busca vetorial
	var vectorRank []string
	vecLookup := make(map[string]rag.SearchResult)
	for _, res := range vecResults {
		vectorRank = append(vectorRank, res.Document.ID)
		vecLookup[res.Document.ID] = res
	}

	// 2. Extrai DocIDs do grafo ordenados por frequência (aresta relacionada)
	docCounts := make(map[string]int)
	for _, rel := range graphResults {
		if rel.DocID != "" {
			docCounts[rel.DocID]++
		}
	}

	type docFreq struct {
		id    string
		count int
	}
	var freqs []docFreq
	for id, count := range docCounts {
		freqs = append(freqs, docFreq{id: id, count: count})
	}
	// Ordena decrescente por ocorrência
	sort.Slice(freqs, func(i, j int) bool {
		return freqs[i].count > freqs[j].count
	})

	var graphRank []string
	for _, f := range freqs {
		graphRank = append(graphRank, f.id)
	}

	// 3. Executa fusão RRF
	rrf := RRF{
		Weights: weights,
		K:       60,
	}
	rankings := map[string][]string{
		"vector": vectorRank,
		"graph":  graphRank,
	}
	merged := rrf.Merge(rankings)

	// 4. Monta o resultado final topK a partir do ranking RRF
	var out []rag.SearchResult
	for i, item := range merged {
		if i >= topK {
			break
		}

		// Tenta reusar do lookup vetorial existente para evitar I/O
		if cached, ok := vecLookup[item.ID]; ok {
			out = append(out, rag.SearchResult{
				Document: cached.Document,
				Score:    float32(item.Score),
			})
			continue
		}

		// Caso contrário, busca do vector store por ID
		doc, err := r.vector.GetDocumentByID(ctx, item.ID)
		if err != nil {
			// Pula se por acaso houver inconsistência (ex: apagado recentemente)
			continue
		}

		out = append(out, rag.SearchResult{
			Document: doc,
			Score:    float32(item.Score),
		})
	}

	return out, nil
}
