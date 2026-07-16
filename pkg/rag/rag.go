// Package rag expõe a API pública do uRag-go: Vector RAG via chromem-go.
package rag

import (
	"context"

	chromem "github.com/philippgille/chromem-go"
)

// Config configura o UnifiedRAG. No MVP cobre apenas o Vector RAG.
type Config struct {
	// EmbeddingProvider: "ollama" ou "openai".
	EmbeddingProvider string
	// EmbeddingModel: nome do modelo no provider escolhido.
	EmbeddingModel string
	// EmbeddingAPIKey: obrigatório apenas para providers hosted (ex: openai).
	EmbeddingAPIKey string
	// EmbeddingBaseURL: override do endpoint Ollama. "" = default
	// (localhost:11434) — precisa ser setado ao rodar contra um Ollama remoto
	// ou em outro host de rede (ex: container "ollama" num docker-compose).
	EmbeddingBaseURL string
	// PersistPath: caminho do arquivo de persistência. Vazio = in-memory.
	PersistPath string
	// Index: "" (exaustivo, default) ou "hnsw" (aproximado, via github.com/coder/hnsw).
	// hnsw não é combinável com PersistPath — o índice ANN não é persistido, então
	// um restart o perderia silenciosamente; New retorna erro nesse caso.
	Index string
}

// UnifiedRAG é o ponto de entrada da biblioteca.
type UnifiedRAG struct {
	vector *vectorStore
}

// New cria um UnifiedRAG a partir de Config.
func New(cfg Config) (*UnifiedRAG, error) {
	v, err := newVectorStore(cfg)
	if err != nil {
		return nil, err
	}
	return &UnifiedRAG{vector: v}, nil
}

// NewWithEmbedding cria um UnifiedRAG com um EmbeddingFunc já pronto, sem passar
// pelo provider resolvido a partir de Config.EmbeddingProvider/EmbeddingModel —
// útil para embedding funcs customizados (provider fora de "ollama"/"openai") ou
// para injetar um fake em testes.
func NewWithEmbedding(cfg Config, embeddingFunc chromem.EmbeddingFunc) (*UnifiedRAG, error) {
	v, err := newVectorStoreWithEmbedding(cfg, embeddingFunc)
	if err != nil {
		return nil, err
	}
	return &UnifiedRAG{vector: v}, nil
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
// conteúdo do documento (whereDocument). Chaves aceitas em whereDocument: "$contains",
// "$not_contains" — semântica definida pelo chromem-go, não pelo uRag-go.
func (u *UnifiedRAG) QueryFiltered(ctx context.Context, question string, topK int, where, whereDocument map[string]string) ([]SearchResult, error) {
	return u.vector.query(ctx, question, topK, where, whereDocument)
}
