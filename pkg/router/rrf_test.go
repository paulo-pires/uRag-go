package router

import (
	"testing"
)

func TestRRFMerge(t *testing.T) {
	rrf := RRF{
		Weights: map[string]float64{
			"vector": 1.0,
			"graph":  1.0,
		},
		K: 60,
	}

	rankings := map[string][]string{
		"vector": {"docA", "docB", "docC"},
		"graph":  {"docB", "docA"},
	}

	merged := rrf.Merge(rankings)

	if len(merged) != 3 {
		t.Fatalf("esperava 3 itens fundidos, obtido %d", len(merged))
	}

	// docA RRF: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01612 = 0.03251
	// docB RRF: 1/(60+2) + 1/(60+1) = 0.01612 + 0.01639 = 0.03251
	// docC RRF: 1/(60+3) = 0.01587
	// Como docA e docB empatam, a ordem exata depende de sort.Slice estável ou do mapa,
	// mas com certeza docC deve ser o último!
	if merged[2].ID != "docC" {
		t.Errorf("esperava docC em último lugar, obtido: %s", merged[2].ID)
	}

	// Testa pesos diferentes
	rrfWeighted := RRF{
		Weights: map[string]float64{
			"vector": 1.0,
			"graph":  2.0, // grafo vale o dobro!
		},
		K: 60,
	}

	// docA RRF: 1*1/(60+1) + 2*1/(60+2) = 0.01639 + 0.03225 = 0.04864
	// docB RRF: 1*1/(60+2) + 2*1/(60+1) = 0.01612 + 0.03278 = 0.04890 (docB vence!)
	mergedWeighted := rrfWeighted.Merge(rankings)

	if mergedWeighted[0].ID != "docB" {
		t.Errorf("esperava docB em primeiro lugar com pesos diferenciados, obtido: %s", mergedWeighted[0].ID)
	}
	if mergedWeighted[1].ID != "docA" {
		t.Errorf("esperava docA em segundo lugar, obtido: %s", mergedWeighted[1].ID)
	}
}
