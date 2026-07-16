// Package graph implementa o Graph RAG do uRag-go: extração de entidades/relações
// via LLM e busca multi-hop sobre o grafo resultante.
package graph

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"urag-go/pkg/rag"
)

// Entity é um nó do grafo.
type Entity struct {
	Name string
	Type string // livre, o que o LLM extrair (ex: "Pessoa", "Empresa")
}

// Relation é uma aresta do grafo, com proveniência do documento de origem.
type Relation struct {
	Source   string // nome normalizado da entidade origem
	Target   string // nome normalizado da entidade destino
	Relation string // texto livre extraído pelo LLM (ex: "trabalha_em")
	DocID    string
}

// Config configura o GraphStore.
type Config struct {
	LLMProvider string // "ollama" ou "openai" (compatível: OpenAI e providers que implementem o mesmo formato)
	LLMModel    string
	// LLMBaseURL: override do endpoint. "" = default do provider (Ollama local,
	// ou api.openai.com para "openai") — obrigatório apontar pra providers
	// OpenAI-compatíveis que não são a OpenAI oficial (vLLM, LM Studio, etc).
	LLMBaseURL string
	// LLMAPIKey: usado só quando LLMProvider="openai".
	LLMAPIKey string
}

// GraphStore é o ponto de entrada do Graph RAG.
type GraphStore struct {
	entities  map[string]Entity
	relations []Relation
	adjacency map[string][]int // nome normalizado -> índices em relations (ambas direções)
	complete  completionFunc
}

// New cria um GraphStore a partir de Config.
func New(cfg Config) (*GraphStore, error) {
	switch cfg.LLMProvider {
	case "ollama":
		return newGraphStoreWithCompletion(ollamaCompletion(cfg.LLMModel, cfg.LLMBaseURL)), nil
	case "openai":
		return newGraphStoreWithCompletion(openaiCompletion(cfg.LLMModel, cfg.LLMBaseURL, cfg.LLMAPIKey)), nil
	default:
		return nil, fmt.Errorf("graph: llm provider desconhecido: %q", cfg.LLMProvider)
	}
}

// NewWithCompletion cria um GraphStore com uma função de completion já pronta,
// sem resolver via Config.LLMProvider/LLMModel — útil para completions
// customizadas (provider fora de "ollama") ou para injetar um fake em testes.
func NewWithCompletion(complete func(ctx context.Context, prompt string) (string, error)) *GraphStore {
	return newGraphStoreWithCompletion(complete)
}

// newGraphStoreWithCompletion permite injetar um completionFunc fake em testes,
// sem depender de Ollama rodando.
func newGraphStoreWithCompletion(complete completionFunc) *GraphStore {
	return &GraphStore{
		entities:  map[string]Entity{},
		adjacency: map[string][]int{},
		complete:  complete,
	}
}

func normalizeName(s string) string {
	return strings.ToLower(strings.TrimSpace(s))
}

type extraction struct {
	Entities []struct {
		Name string `json:"name"`
		Type string `json:"type"`
	} `json:"entities"`
	Relations []struct {
		Source   string `json:"source"`
		Target   string `json:"target"`
		Relation string `json:"relation"`
	} `json:"relations"`
}

const extractPrompt = `Extraia entidades e relações do texto abaixo. Responda apenas em JSON:
{"entities":[{"name":"...","type":"..."}],"relations":[{"source":"...","target":"...","relation":"..."}]}

Texto: %s`

// AddDocuments extrai entidades/relações de cada documento (1 chamada LLM por
// documento, sem chunking) e funde no grafo em memória. Um documento cujo LLM
// devolva JSON inválido é descartado sem travar o restante do lote.
func (g *GraphStore) AddDocuments(ctx context.Context, docs []rag.Document) error {
	for _, doc := range docs {
		raw, err := g.complete(ctx, fmt.Sprintf(extractPrompt, doc.Content))
		if err != nil {
			return fmt.Errorf("graph: extrair doc %s: %w", doc.ID, err)
		}

		var ext extraction
		if err := json.Unmarshal([]byte(raw), &ext); err != nil {
			continue
		}

		g.merge(doc.ID, ext)
	}
	return nil
}

func (g *GraphStore) merge(docID string, ext extraction) {
	for _, e := range ext.Entities {
		key := normalizeName(e.Name)
		if key == "" {
			continue
		}
		if _, exists := g.entities[key]; !exists {
			g.entities[key] = Entity{Name: e.Name, Type: e.Type}
		}
	}

	for _, r := range ext.Relations {
		source := normalizeName(r.Source)
		target := normalizeName(r.Target)
		if source == "" || target == "" {
			continue
		}
		rel := Relation{Source: source, Target: target, Relation: r.Relation, DocID: docID}
		idx := len(g.relations)
		g.relations = append(g.relations, rel)
		g.adjacency[source] = append(g.adjacency[source], idx)
		g.adjacency[target] = append(g.adjacency[target], idx)
	}
}

// Query acha entidades cujo nome aparece na pergunta (seed) e expande por BFS
// até hops relações de distância, devolvendo as relações tocadas como contexto.
// Não sintetiza resposta em texto — isso é responsabilidade de uma camada futura.
func (g *GraphStore) Query(_ context.Context, question string, hops int) ([]Relation, error) {
	q := normalizeName(question)

	frontier := map[string]bool{}
	for name := range g.entities {
		if name != "" && strings.Contains(q, name) {
			frontier[name] = true
		}
	}

	visited := map[string]bool{}
	for name := range frontier {
		visited[name] = true
	}

	seenRelation := map[int]bool{}
	var result []Relation

	for hop := 0; hop < hops && len(frontier) > 0; hop++ {
		next := map[string]bool{}
		for name := range frontier {
			for _, idx := range g.adjacency[name] {
				if seenRelation[idx] {
					continue
				}
				seenRelation[idx] = true
				rel := g.relations[idx]
				result = append(result, rel)

				for _, neighbor := range []string{rel.Source, rel.Target} {
					if !visited[neighbor] {
						visited[neighbor] = true
						next[neighbor] = true
					}
				}
			}
		}
		frontier = next
	}

	return result, nil
}
