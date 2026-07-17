// Package graph implementa o Graph RAG do uRag-go: extração de entidades/relações
// via LLM e busca multi-hop sobre o grafo resultante.
package graph

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"urag-go/pkg/graph/persist"
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
	// PersistDSN: DSN para persistência do grafo (ex: "file:graph.db")
	PersistDSN string
	// PersistEnabled: habilita persistência
	PersistEnabled bool
	// CacheSize: tamanho do cache em memória (default: 1000)
	CacheSize int
	// CacheTTL: TTL do cache (default: 5 minutos)
	CacheTTL time.Duration
}

// GraphStore é o ponto de entrada do Graph RAG.
type GraphStore struct {
	entities   map[string]Entity
	relations  []Relation
	adjacency  map[string][]int // nome normalizado -> índices em relations (ambas direções)
	complete   completionFunc
	store      *persist.Store // persistência opcional
	config     Config
	entityID   map[string]string // nome normalizado -> ID persistido
	relationID map[int]string    // índice -> ID persistido
}

// New cria um GraphStore a partir de Config.
func New(cfg Config) (*GraphStore, error) {
	// Valores padrão para cache
	if cfg.CacheSize == 0 {
		cfg.CacheSize = 1000
	}
	if cfg.CacheTTL == 0 {
		cfg.CacheTTL = 5 * time.Minute
	}

	var store *persist.Store
	var err error

	// Inicializa persistência se habilitada
	if cfg.PersistEnabled && cfg.PersistDSN != "" {
		store, err = persist.NewStore(persist.Config{
			DSN:       cfg.PersistDSN,
			CacheSize: cfg.CacheSize,
			CacheTTL:  cfg.CacheTTL,
		})
		if err != nil {
			return nil, fmt.Errorf("graph: init persist: %w", err)
		}
	}

	g := &GraphStore{
		entities:   map[string]Entity{},
		adjacency:  map[string][]int{},
		config:     cfg,
		store:      store,
		entityID:   map[string]string{},
		relationID: map[int]string{},
	}

	// Define a função de completion baseada no provider
	switch cfg.LLMProvider {
	case "ollama":
		g.complete = ollamaCompletion(cfg.LLMModel, cfg.LLMBaseURL)
	case "openai":
		g.complete = openaiCompletion(cfg.LLMModel, cfg.LLMBaseURL, cfg.LLMAPIKey)
	default:
		return nil, fmt.Errorf("graph: llm provider desconhecido: %q", cfg.LLMProvider)
	}

	// Carrega grafo existente se persistência estiver habilitada
	if store != nil {
		if err := g.loadFromStore(context.Background()); err != nil {
			return nil, fmt.Errorf("graph: load from store: %w", err)
		}
	}

	return g, nil
}

// NewWithCompletion cria um GraphStore com uma função de completion já pronta,
// sem resolver via Config.LLMProvider/LLMModel — útil para completions
// customizadas (provider fora de "ollama") ou para injetar um fake em testes.
func NewWithCompletion(complete func(ctx context.Context, prompt string) (string, error)) *GraphStore {
	return newGraphStoreWithCompletion(complete, persist.Config{})
}

// NewWithCompletionAndPersist cria um GraphStore com completion e persistência
func NewWithCompletionAndPersist(
	complete func(ctx context.Context, prompt string) (string, error),
	persistCfg persist.Config,
) (*GraphStore, error) {
	g := newGraphStoreWithCompletion(complete, persistCfg)

	// Inicializa persistência
	if persistCfg.DSN != "" {
		store, err := persist.NewStore(persistCfg)
		if err != nil {
			return nil, fmt.Errorf("graph: init persist: %w", err)
		}
		g.store = store

		if err := g.loadFromStore(context.Background()); err != nil {
			return nil, fmt.Errorf("graph: load from store: %w", err)
		}
	}

	return g, nil
}

// newGraphStoreWithCompletion permite injetar um completionFunc fake em testes,
// sem depender de Ollama rodando.
func newGraphStoreWithCompletion(complete completionFunc, persistCfg persist.Config) *GraphStore {
	return &GraphStore{
		entities:   map[string]Entity{},
		adjacency:  map[string][]int{},
		complete:   complete,
		entityID:   map[string]string{},
		relationID: map[int]string{},
	}
}

// Close fecha o GraphStore e suas conexões
func (g *GraphStore) Close() error {
	if g.store != nil {
		return g.store.Close()
	}
	return nil
}

// loadFromStore carrega o grafo do storage
func (g *GraphStore) loadFromStore(ctx context.Context) error {
	snapshot, err := g.store.LoadFullGraph(ctx)
	if err != nil {
		return err
	}

	idToNormalizedName := make(map[string]string)

	// Carrega entidades
	for _, entity := range snapshot.Entities {
		normalized := normalizeName(entity.Name)
		g.entities[normalized] = Entity{
			Name: entity.Name,
			Type: entity.Type,
		}
		g.entityID[normalized] = entity.ID
		idToNormalizedName[entity.ID] = normalized
	}

	// Carrega relações
	for _, rel := range snapshot.Relations {
		idx := len(g.relations)
		sourceName := idToNormalizedName[rel.SourceID]
		targetName := idToNormalizedName[rel.TargetID]
		if sourceName == "" || targetName == "" {
			continue // Segurança caso haja dados órfãos
		}

		docID := ""
		if rel.Properties != nil {
			if d, ok := rel.Properties["doc_id"].(string); ok {
				docID = d
			}
		}

		g.relations = append(g.relations, Relation{
			Source:   sourceName,
			Target:   targetName,
			Relation: rel.Type,
			DocID:    docID,
		})
		g.relationID[idx] = rel.ID
		g.adjacency[sourceName] = append(g.adjacency[sourceName], idx)
		g.adjacency[targetName] = append(g.adjacency[targetName], idx)
	}

	return nil
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

		if err := g.mergeWithPersist(ctx, doc.ID, ext); err != nil {
			return fmt.Errorf("graph: merge doc %s: %w", doc.ID, err)
		}
	}
	return nil
}

// mergeWithPersist funde as extrações e persiste se habilitado
func (g *GraphStore) mergeWithPersist(ctx context.Context, docID string, ext extraction) error {
	// Se persistência estiver habilitada, salva no SQLite
	if g.store != nil {
		// Salva entidades
		for _, e := range ext.Entities {
			normalized := normalizeName(e.Name)
			if normalized == "" {
				continue
			}

			// Verifica se já existe
			if _, exists := g.entityID[normalized]; !exists {
				entity := &persist.Entity{
					ID:   fmt.Sprintf("ent_%d", time.Now().UnixNano()),
					Name: e.Name,
					Type: e.Type,
					Properties: map[string]interface{}{
						"normalized": normalized,
					},
				}
				if err := g.store.AddEntity(ctx, entity); err != nil {
					return fmt.Errorf("salvar entidade: %w", err)
				}
				g.entityID[normalized] = entity.ID
			}
		}

		// Salva relações
		for _, r := range ext.Relations {
			source := normalizeName(r.Source)
			target := normalizeName(r.Target)
			if source == "" || target == "" {
				continue
			}

			// Verifica se a relação já existe
			exists := false
			for _, rel := range g.relations {
				if rel.Source == source && rel.Target == target && rel.Relation == r.Relation && rel.DocID == docID {
					exists = true
					break
				}
			}

			if !exists {
				rel := &persist.Relation{
					ID:       fmt.Sprintf("rel_%d", time.Now().UnixNano()),
					SourceID: g.entityID[source],
					TargetID: g.entityID[target],
					Type:     r.Relation,
					Properties: map[string]interface{}{
						"doc_id": docID,
					},
				}
				if err := g.store.AddRelation(ctx, rel); err != nil {
					return fmt.Errorf("salvar relação: %w", err)
				}
			}
		}
	}

	// Por fim, faz o merge em memória
	g.merge(docID, ext)

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

// GetStats retorna estatísticas do grafo
func (g *GraphStore) GetStats() map[string]interface{} {
	stats := map[string]interface{}{
		"entities":  len(g.entities),
		"relations": len(g.relations),
	}

	if g.store != nil {
		stats["persisted"] = true
		stats["dsn"] = g.config.PersistDSN
	} else {
		stats["persisted"] = false
	}

	return stats
}

// LoadFullGraph carrega o grafo completo do storage
func (g *GraphStore) LoadFullGraph(ctx context.Context) (*persist.GraphSnapshot, error) {
	if g.store == nil {
		return nil, fmt.Errorf("graph: persistência não habilitada")
	}
	return g.store.LoadFullGraph(ctx)
}

// SaveFullGraph salva o grafo completo no storage
func (g *GraphStore) SaveFullGraph(ctx context.Context, snapshot *persist.GraphSnapshot) error {
	if g.store == nil {
		return fmt.Errorf("graph: persistência não habilitada")
	}
	return g.store.SaveFullGraph(ctx, snapshot)
}
