// Package rag expõe a API pública do uRag-go: Vector RAG via chromem-go.
package rag

import (
	"context"
	"fmt"
	"hash/fnv"
	"time"

	chromem "github.com/philippgille/chromem-go"
)

// Config configura o UnifiedRAG.
type Config struct {
	// EmbeddingProvider: "ollama" ou "openai".
	EmbeddingProvider string
	// EmbeddingModel: nome do modelo no provider escolhido.
	EmbeddingModel string
	// EmbeddingAPIKey: obrigatório apenas para providers hosted (ex: openai).
	EmbeddingAPIKey string
	// EmbeddingBaseURL: override do endpoint Ollama. "" = default (localhost:11434).
	EmbeddingBaseURL string
	// PersistPath: persistência do chromem-go. Vazio = in-memory. Ignorado para VectorBackend=qdrant.
	PersistPath string
	// Index: "" (exaustivo, default) ou "hnsw". Ignorado para VectorBackend=qdrant.
	Index string
	// CacheSize: tamanho máximo do cache de embeddings (default: 1000).
	CacheSize int
	// CacheTTL: TTL dos itens no cache (default: 5 minutos).
	CacheTTL time.Duration
	// VectorBackend: "" ou "chromem" (default) | "qdrant".
	VectorBackend string
	// QdrantURL: endpoint REST do Qdrant (default "http://localhost:6333").
	QdrantURL string
	// QdrantCollection: nome da collection no Qdrant (default "documents").
	QdrantCollection string
	// QdrantAPIKey: API key para Qdrant Cloud. "" = sem autenticação.
	QdrantAPIKey string
}

// vectorBackend é a interface interna de UnifiedRAG.
// Implementada por *vectorStore (chromem) e *qdrantStore.
type vectorBackend interface {
	add(ctx context.Context, docs []Document) error
	query(ctx context.Context, question string, topK int, where, whereDocument map[string]string) ([]SearchResult, error)
	getByID(ctx context.Context, id string) (Document, error)
	generateEmbedding(ctx context.Context, text string) ([]float32, error)
}

// UnifiedRAG é o ponto de entrada da biblioteca.
type UnifiedRAG struct {
	vector vectorBackend
	config Config
	cache  *EmbeddingCache
}

// New cria um UnifiedRAG a partir de Config.
func New(cfg Config) (*UnifiedRAG, error) {
	if cfg.CacheSize <= 0 {
		cfg.CacheSize = 1000
	}
	if cfg.CacheTTL <= 0 {
		cfg.CacheTTL = 5 * time.Minute
	}

	cache := NewEmbeddingCache(cfg.CacheSize, cfg.CacheTTL)

	var v vectorBackend
	var err error
	if cfg.VectorBackend == "qdrant" {
		embFn, embErr := embeddingFuncFor(cfg)
		if embErr != nil {
			return nil, embErr
		}
		v = newQdrantStore(cfg, wrapWithCache(cache, embFn))
	} else {
		v, err = newVectorStore(cfg, cache)
		if err != nil {
			return nil, err
		}
	}
	return &UnifiedRAG{vector: v, config: cfg, cache: cache}, nil
}

// NewWithEmbedding cria um UnifiedRAG com um EmbeddingFunc já pronto — útil para
// injetar um fake em testes ou usar um provider customizado.
// Apenas o backend chromem é suportado neste path; para Qdrant use New().
func NewWithEmbedding(cfg Config, embeddingFunc chromem.EmbeddingFunc) (*UnifiedRAG, error) {
	if cfg.CacheSize <= 0 {
		cfg.CacheSize = 1000
	}
	if cfg.CacheTTL <= 0 {
		cfg.CacheTTL = 5 * time.Minute
	}
	cache := NewEmbeddingCache(cfg.CacheSize, cfg.CacheTTL)

	v, err := newVectorStoreWithEmbedding(cfg, wrapWithCache(cache, embeddingFunc))
	if err != nil {
		return nil, err
	}
	return &UnifiedRAG{vector: v, config: cfg, cache: cache}, nil
}

// AddDocuments indexa documentos no Vector RAG.
func (u *UnifiedRAG) AddDocuments(ctx context.Context, docs []Document) error {
	return u.vector.add(ctx, docs)
}

// Query busca os topK documentos mais relevantes para a pergunta.
func (u *UnifiedRAG) Query(ctx context.Context, question string, topK int) ([]SearchResult, error) {
	return u.vector.query(ctx, question, topK, nil, nil)
}

// QueryFiltered é como Query, mas restringe os resultados por metadata (where) e/ou
// conteúdo do documento (whereDocument). Chaves aceitas in whereDocument: "$contains",
// "$not_contains" — semântica definida pelo chromem-go, não pelo uRag-go.
func (u *UnifiedRAG) QueryFiltered(ctx context.Context, question string, topK int, where, whereDocument map[string]string) ([]SearchResult, error) {
	return u.vector.query(ctx, question, topK, where, whereDocument)
}

// Close fecha o UnifiedRAG e libera recursos
func (r *UnifiedRAG) Close() error {
	// O chromem-go Collection não tem Close explícito
	// Mas podemos limpar o cache e forçar persistência
	if r.cache != nil {
		r.cache.Clear()
	}

	// Se houver persistência, chromem-go já salva automaticamente
	// em cada operação, mas podemos garantir que o último estado foi salvo
	if r.config.PersistPath != "" {
		// chromem-go persiste automaticamente, não precisa de ação extra
		// apenas retornamos nil
	}

	return nil
}

// GetDocumentByID recupera um documento específico pelo seu ID.
func (u *UnifiedRAG) GetDocumentByID(ctx context.Context, id string) (Document, error) {
	return u.vector.getByID(ctx, id)
}

// GenerateEmbedding gera o vetor numérico (embedding) para o texto fornecido.
func (u *UnifiedRAG) GenerateEmbedding(ctx context.Context, text string) ([]float32, error) {
	return u.vector.generateEmbedding(ctx, text)
}

func wrapWithCache(cache *EmbeddingCache, baseFunc chromem.EmbeddingFunc) chromem.EmbeddingFunc {
	return func(ctx context.Context, text string) ([]float32, error) {
		if cache == nil {
			return baseFunc(ctx, text)
		}
		h := fnv.New64a()
		h.Write([]byte(text))
		key := fmt.Sprintf("%x", h.Sum64())

		if vec, hit := cache.Get(key); hit {
			return vec, nil
		}

		vec, err := baseFunc(ctx, text)
		if err != nil {
			return nil, err
		}

		cache.Set(key, vec)
		return vec, nil
	}
}
