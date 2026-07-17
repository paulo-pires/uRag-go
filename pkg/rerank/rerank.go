package rerank

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"

	"urag-go/internal/ollama"
	"urag-go/internal/openai"
)

type Result struct {
	DocumentID string
	Content    string
	Score      float32
}

type ReRanker struct {
	provider string
	model    string
	baseURL  string
	apiKey   string
}

// New cria um novo re-ranker baseado em LLM (LLM-as-a-Judge).
func New(provider, model, baseURL, apiKey string) *ReRanker {
	if model == "" {
		model = "granite4:micro-h" // fallback
	}
	return &ReRanker{
		provider: provider,
		model:    model,
		baseURL:  baseURL,
		apiKey:   apiKey,
	}
}

// ReRank re-ordena os resultados de busca iniciais pontuando a relevância
// de cada trecho de documento individualmente através de chamadas concorrentes ao LLM.
func (r *ReRanker) ReRank(ctx context.Context, query string, results []Result) ([]Result, error) {
	if len(results) == 0 {
		return results, nil
	}

	var wg sync.WaitGroup
	scores := make([]float32, len(results))
	errs := make([]error, len(results))

	for idx, res := range results {
		wg.Add(1)
		go func(i int, content string) {
			defer wg.Done()

			score, err := r.evaluateRelevancy(ctx, query, content)
			if err != nil {
				errs[i] = err
				scores[i] = 0.0 // fallback seguro em caso de erro da API
				return
			}
			scores[i] = score
		}(idx, res.Content)
	}

	wg.Wait()

	// Opcional: checar erros sérios de rede, mas priorizamos robustez e tolerância a falhas parciais
	// de modo que se uma chamada falhar, o documento fica com score zero no fim da fila.

	// Atualiza os scores nos resultados
	reranked := make([]Result, len(results))
	for i, res := range results {
		reranked[i] = Result{
			DocumentID: res.DocumentID,
			Content:    res.Content,
			Score:      scores[i],
		}
	}

	// Ordena decrescente por score
	sort.Slice(reranked, func(i, j int) bool {
		if reranked[i].Score == reranked[j].Score {
			// preserva ordem original em caso de empate (estabilidade)
			return i < j
		}
		return reranked[i].Score > reranked[j].Score
	})

	return reranked, nil
}

func (r *ReRanker) evaluateRelevancy(ctx context.Context, query, content string) (float32, error) {
	prompt := fmt.Sprintf(`Avalie de 0 a 10 a relevância do trecho de documento abaixo para responder à pergunta fornecida, onde:
- 0: O trecho é totalmente irrelevante.
- 10: O trecho responde perfeitamente ou contém informações essenciais.

Pergunta: %s
Trecho: %s

Responda apenas com o número inteiro da nota (ex: 7). Não dê explicações.
Nota:`, query, content)

	var answer string
	var err error

	if r.provider == "openai" {
		answer, err = openai.Complete(ctx, r.baseURL, r.apiKey, r.model, prompt, false)
	} else {
		answer, err = ollama.Complete(ctx, r.baseURL, r.model, prompt, false)
	}

	if err != nil {
		return 0, err
	}

	// Limpa resposta do LLM para ler apenas a nota numérica
	cleanAnswer := strings.TrimSpace(answer)
	// Se retornar texto contendo números, tenta extrair o primeiro dígito
	var scoreVal int
	_, err = fmt.Sscanf(cleanAnswer, "%d", &scoreVal)
	if err != nil {
		// Fallback simples caso venha texto extra no início
		// tenta converter caractere a caractere
		for _, char := range cleanAnswer {
			if char >= '0' && char <= '9' {
				scoreVal, _ = strconv.Atoi(string(char))
				break
			}
		}
	}

	if scoreVal < 0 {
		scoreVal = 0
	}
	if scoreVal > 10 {
		scoreVal = 10
	}

	// Normaliza entre 0.0 e 1.0
	return float32(scoreVal) / 10.0, nil
}