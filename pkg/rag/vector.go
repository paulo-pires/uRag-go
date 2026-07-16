package rag

import (
	"context"
	"fmt"

	"urag-go/internal/hnsw"
	chromem "github.com/philippgille/chromem-go"
)

const collectionName = "documents"

// vectorStore embrulha o *chromem.DB — não exposto no contrato público de UnifiedRAG.
type vectorStore struct {
	db            *chromem.DB
	collection    *chromem.Collection
	embeddingFunc chromem.EmbeddingFunc
	ann           *hnsw.Graph[string] // nil quando Config.Index != "hnsw"
}

func newVectorStore(cfg Config) (*vectorStore, error) {
	embeddingFunc, err := embeddingFuncFor(cfg)
	if err != nil {
		return nil, err
	}
	return newVectorStoreWithEmbedding(cfg, embeddingFunc)
}

// newVectorStoreWithEmbedding permite injetar um EmbeddingFunc fake em testes,
// sem depender de um provider externo (Ollama/OpenAI) rodando.
func newVectorStoreWithEmbedding(cfg Config, embeddingFunc chromem.EmbeddingFunc) (*vectorStore, error) {
	switch cfg.Index {
	case "", "hnsw":
	default:
		return nil, fmt.Errorf("rag: index desconhecido: %q", cfg.Index)
	}
	if cfg.Index == "hnsw" && cfg.PersistPath != "" {
		return nil, fmt.Errorf("rag: Index=hnsw não é combinável com PersistPath (índice ANN não é persistido)")
	}

	var db *chromem.DB
	var err error
	if cfg.PersistPath != "" {
		db, err = chromem.NewPersistentDB(cfg.PersistPath, true)
		if err != nil {
			return nil, fmt.Errorf("rag: abrir db persistente: %w", err)
		}
	} else {
		db = chromem.NewDB()
	}

	collection, err := db.GetOrCreateCollection(collectionName, nil, embeddingFunc)
	if err != nil {
		return nil, fmt.Errorf("rag: criar collection: %w", err)
	}

	var ann *hnsw.Graph[string]
	if cfg.Index == "hnsw" {
		ann = hnsw.NewGraph[string]()
	}

	return &vectorStore{db: db, collection: collection, embeddingFunc: embeddingFunc, ann: ann}, nil
}

func embeddingFuncFor(cfg Config) (chromem.EmbeddingFunc, error) {
	switch cfg.EmbeddingProvider {
	case "ollama":
		model := cfg.EmbeddingModel
		if model == "" {
			model = "nomic-embed-text"
		}
		return chromem.NewEmbeddingFuncOllama(model, cfg.EmbeddingBaseURL), nil
	case "openai":
		return chromem.NewEmbeddingFuncOpenAI(cfg.EmbeddingAPIKey, chromem.EmbeddingModelOpenAI(cfg.EmbeddingModel)), nil
	default:
		return nil, fmt.Errorf("rag: embedding provider desconhecido: %q", cfg.EmbeddingProvider)
	}
}

func (v *vectorStore) add(ctx context.Context, docs []Document) error {
	chromemDocs := make([]chromem.Document, len(docs))
	for i, d := range docs {
		meta := d.Meta
		if meta == nil {
			meta = map[string]string{}
		}
		if d.Source != "" {
			meta["source"] = d.Source
		}
		chromemDocs[i] = chromem.Document{
			ID:       d.ID,
			Content:  d.Content,
			Metadata: meta,
		}

		if v.ann != nil {
			vec, err := v.embeddingFunc(ctx, d.Content)
			if err != nil {
				return fmt.Errorf("rag: gerar embedding para índice ann (doc %s): %w", d.ID, err)
			}
			chromemDocs[i].Embedding = vec
			v.ann.Add(hnsw.MakeNode(d.ID, hnsw.Vector(vec)))
		}
	}
	return v.collection.AddDocuments(ctx, chromemDocs, 1)
}

func (v *vectorStore) query(ctx context.Context, question string, topK int, where, whereDocument map[string]string) ([]SearchResult, error) {
	if topK <= 0 {
		return nil, nil
	}
	if v.ann != nil {
		return v.queryANN(ctx, question, topK, where, whereDocument)
	}
	return v.queryExhaustive(ctx, question, topK, where, whereDocument)
}

func (v *vectorStore) queryExhaustive(ctx context.Context, question string, topK int, where, whereDocument map[string]string) ([]SearchResult, error) {
	if topK > v.collection.Count() {
		topK = v.collection.Count()
	}
	if topK == 0 {
		return nil, nil
	}

	results, err := v.collection.Query(ctx, question, topK, where, whereDocument)
	if err != nil {
		return nil, fmt.Errorf("rag: query: %w", err)
	}

	out := make([]SearchResult, len(results))
	for i, r := range results {
		out[i] = SearchResult{
			Document: Document{
				ID:      r.ID,
				Content: r.Content,
				Source:  r.Metadata["source"],
				Meta:    r.Metadata,
			},
			Score: r.Similarity,
		}
	}
	return out, nil
}

// queryANN busca no índice hnsw (aproximado). whereDocument não é suportado nesse
// caminho — coder/hnsw não indexa conteúdo, só embeddings; where é aplicado como
// filtro pós-busca sobre a metadata de cada resultado.
func (v *vectorStore) queryANN(ctx context.Context, question string, topK int, where, whereDocument map[string]string) ([]SearchResult, error) {
	if len(whereDocument) > 0 {
		return nil, fmt.Errorf("rag: whereDocument não é suportado com Index=hnsw")
	}

	queryVec, err := v.embeddingFunc(ctx, question)
	if err != nil {
		return nil, fmt.Errorf("rag: gerar embedding da pergunta: %w", err)
	}

	nodes := v.ann.Search(queryVec, topK)
	out := make([]SearchResult, 0, len(nodes))
	for _, n := range nodes {
		doc, err := v.collection.GetByID(ctx, n.Key)
		if err != nil {
			continue // inconsistência ann/collection (ex: doc removido) — pula
		}
		if !matchesWhere(doc.Metadata, where) {
			continue
		}
		out = append(out, SearchResult{
			Document: Document{
				ID:      doc.ID,
				Content: doc.Content,
				Source:  doc.Metadata["source"],
				Meta:    doc.Metadata,
			},
			Score: 1 - hnsw.CosineDistance(queryVec, n.Value),
		})
	}
	return out, nil
}

func matchesWhere(meta, where map[string]string) bool {
	for k, want := range where {
		if meta[k] != want {
			return false
		}
	}
	return true
}
