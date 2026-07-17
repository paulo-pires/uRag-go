// Package mcpserver expõe as 4 stores do uRag-go (vector/graph/tree/sql) como
// tools MCP sobre um único servidor com estado persistente em memória — um
// agente pode chamar *_add numa invocação e *_query em outra, no mesmo processo.
package mcpserver

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"urag-go/pkg/graph"
	"urag-go/pkg/rag"
	urasql "urag-go/pkg/sql"
	"urag-go/pkg/telemetry"
	"urag-go/pkg/tree"
)

// Config configura o servidor MCP.
type Config struct {
	// VectorDBPath: persistência do vector store (chromem-go). "" = in-memory.
	VectorDBPath string
	// EmbeddingProvider: "ollama" (default) ou "openai".
	EmbeddingProvider string
	// EmbeddingModel: nome do modelo no provider de embedding escolhido.
	EmbeddingModel string
	// EmbeddingAPIKey: obrigatório apenas para EmbeddingProvider="openai".
	EmbeddingAPIKey string
	// EmbeddingBaseURL: override do endpoint Ollama de embedding. "" = default
	// (localhost:11434) — precisa ser setado rodando contra Ollama remoto ou
	// noutro host de rede (ex: container "ollama" num docker-compose).
	EmbeddingBaseURL string
	// LLMProvider: "ollama" (default) ou "openai" — usado por graph/tree/sql
	// (extração/navegação/geração).
	LLMProvider string
	// LLMModel: modelo no provider de LLM escolhido.
	LLMModel string
	// LLMBaseURL: override do endpoint de graph/tree/sql. "" = default do
	// provider — obrigatório pra providers OpenAI-compatíveis que não são a
	// OpenAI oficial.
	LLMBaseURL string
	// LLMAPIKey: usado só quando LLMProvider="openai".
	LLMAPIKey string
	// SQLDSN: caminho do banco SQLite. "" = tool sql_query não é registrada.
	SQLDSN string
	// GraphPersist: DSN para persistência do grafo (ex: "file:graph.db")
	GraphPersist string
	// EmbeddingCacheSize: tamanho do cache de embeddings
	EmbeddingCacheSize int
	// EmbeddingCacheTTL: TTL do cache de embeddings
	EmbeddingCacheTTL time.Duration
}

// Server é o servidor MCP do uRag-go, com as 4 stores já instanciadas.
type Server struct {
	mcp    *mcp.Server
	vector *rag.UnifiedRAG
	graph  *graph.GraphStore
	tree   *tree.Tree
	sql    *urasql.Store
	config Config
}

// New cria o Server: instancia vector/graph/tree sempre; sql só se cfg.SQLDSN
// não for vazio (conecta a um banco já populado, mesma decisão do Router).
func New(cfg Config) (*Server, error) {
	embeddingProvider := cfg.EmbeddingProvider
	if embeddingProvider == "" {
		embeddingProvider = "ollama"
	}
	embeddingModel := cfg.EmbeddingModel
	if embeddingModel == "" {
		embeddingModel = "nomic-embed-text"
	}

	vector, err := rag.New(rag.Config{
		EmbeddingProvider: embeddingProvider,
		EmbeddingModel:    embeddingModel,
		EmbeddingAPIKey:   cfg.EmbeddingAPIKey,
		EmbeddingBaseURL:  cfg.EmbeddingBaseURL,
		PersistPath:       cfg.VectorDBPath,
		CacheSize:         cfg.EmbeddingCacheSize,
		CacheTTL:          cfg.EmbeddingCacheTTL,
	})
	if err != nil {
		return nil, fmt.Errorf("mcpserver: iniciar vector: %w", err)
	}

	llmProvider := cfg.LLMProvider
	if llmProvider == "" {
		llmProvider = "ollama"
	}
	llmModel := cfg.LLMModel
	if llmModel == "" {
		llmModel = "granite4:micro-h"
	}

	// Graph RAG com persistência
	graphCfg := graph.Config{
		LLMProvider:    llmProvider,
		LLMModel:       llmModel,
		LLMBaseURL:     cfg.LLMBaseURL,
		LLMAPIKey:      cfg.LLMAPIKey,
		PersistDSN:     cfg.GraphPersist,
		PersistEnabled: cfg.GraphPersist != "",
		CacheSize:      1000,
		CacheTTL:       5 * time.Minute,
	}
	g, err := graph.New(graphCfg)
	if err != nil {
		return nil, fmt.Errorf("mcpserver: iniciar graph: %w", err)
	}

	t, err := tree.New(tree.Config{
		LLMProvider: llmProvider,
		LLMModel:    llmModel,
		LLMBaseURL:  cfg.LLMBaseURL,
		LLMAPIKey:   cfg.LLMAPIKey,
	})
	if err != nil {
		return nil, fmt.Errorf("mcpserver: iniciar tree: %w", err)
	}

	sqlDSN := cfg.SQLDSN
	if sqlDSN == "" {
		sqlDSN = "urag_sql.db"
	}
	sqlStore, err := urasql.New(urasql.Config{
		DSN:         sqlDSN,
		LLMProvider: llmProvider,
		LLMModel:    llmModel,
		LLMBaseURL:  cfg.LLMBaseURL,
		LLMAPIKey:   cfg.LLMAPIKey,
	})
	if err != nil {
		return nil, fmt.Errorf("mcpserver: iniciar sql: %w", err)
	}

	s := &Server{
		mcp:    mcp.NewServer(&mcp.Implementation{Name: "urag-mcp", Version: "v0.1.0"}, nil),
		vector: vector,
		graph:  g,
		tree:   t,
		sql:    sqlStore,
		config: cfg,
	}
	s.registerTools()
	return s, nil
}

// Run serve as tools registradas sobre stdio. Bloqueia até o transporte fechar.
func (s *Server) Run(ctx context.Context) error {
	return s.mcp.Run(ctx, &mcp.StdioTransport{})
}

// RunHTTP serve as tools registradas sobre HTTP (transporte "Streamable HTTP"
// do MCP — usa Server-Sent Events por baixo pra streaming, um único endpoint
// POST/GET). Diferente de Run (stdio, 1 processo por cliente), permite uma
// UI web ou qualquer cliente na rede conectar diretamente. Todas as sessões
// HTTP compartilham a mesma instância de Server (e portanto as mesmas
// stores/estado em memória) — não há isolamento multi-tenant, mesma decisão
// de escopo já registrada no SPEC.md pro transporte stdio. Bloqueia até o
// contexto ser cancelado ou o listener falhar.
func (s *Server) RunHTTP(ctx context.Context, addr string) error {
	mcpHandler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return s.mcp }, nil)

	mux := http.NewServeMux()
	mux.Handle("/", mcpHandler)
	mux.HandleFunc("/mcp/stream", s.handleStream)
	mux.HandleFunc("/metrics", s.handleMetrics)

	httpServer := &http.Server{Addr: addr, Handler: withCORS(mux)}

	errCh := make(chan error, 1)
	go func() { errCh <- httpServer.ListenAndServe() }()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		return httpServer.Shutdown(context.Background())
	}
}

// withCORS libera acesso de qualquer origem — necessário pra uma UI web
// (rodando num domínio/porta diferente, ex: localhost:5173 no Vite dev
// server) conseguir chamar esse endpoint via fetch/EventSource do
// navegador. Sem autenticação embutida (mesma postura do resto do
// transporte HTTP, ver SPEC.md Fase 8) — coloque atrás de um proxy que
// autentique se for expor além de localhost.
func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Mcp-Session-Id, Mcp-Protocol-Version")
		w.Header().Set("Access-Control-Expose-Headers", "Mcp-Session-Id")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// Close libera as conexões
func (s *Server) Close() error {
	var errs []string

	if s.vector != nil {
		if err := s.vector.Close(); err != nil {
			errs = append(errs, fmt.Sprintf("vector: %v", err))
		}
	}

	if s.graph != nil {
		if err := s.graph.Close(); err != nil {
			errs = append(errs, fmt.Sprintf("graph: %v", err))
		}
	}

	// tree não tem Close, apenas ignorar

	if s.sql != nil {
		if err := s.sql.Close(); err != nil {
			errs = append(errs, fmt.Sprintf("sql: %v", err))
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("close errors: %s", errs)
	}
	return nil
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(telemetry.GlobalCollector.RenderPrometheus()))
}
