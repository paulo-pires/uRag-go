package router

import "sort"

type RRFResult struct {
	ID    string
	Score float64
}

type RRF struct {
	Weights map[string]float64
	K       int // constante de suavização (default: 60)
}

func (r *RRF) Merge(rankings map[string][]string) []RRFResult {
	k := r.K
	if k <= 0 {
		k = 60 // default do RRF
	}

	scores := make(map[string]float64)

	for store, ranking := range rankings {
		weight := 1.0
		if w, ok := r.Weights[store]; ok {
			weight = w
		}

		for idx, docID := range ranking {
			// rank é 1-indexado
			rank := idx + 1
			scores[docID] += weight * (1.0 / float64(k+rank))
		}
	}

	var merged []RRFResult
	for docID, score := range scores {
		merged = append(merged, RRFResult{
			ID:    docID,
			Score: score,
		})
	}

	// Ordena por score decrescente
	sort.Slice(merged, func(i, j int) bool {
		return merged[i].Score > merged[j].Score
	})

	return merged
}
