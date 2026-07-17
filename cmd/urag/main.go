// Command urag é a CLI do uRag-go: add/query sobre Vector RAG (persistente),
// e comandos "ask" single-shot para graph/tree/router (in-memory, sem
// persistência — carregam e consultam na mesma invocação).
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"urag-go/pkg/eval"
	"urag-go/pkg/graph"
	"urag-go/pkg/mcpserver"
	"urag-go/pkg/rag"
	"urag-go/pkg/router"
	urasql "urag-go/pkg/sql"
	"urag-go/pkg/telemetry"
	"urag-go/pkg/tree"
)

func main() {
	loadEnv()
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "add":
		runAdd(os.Args[2:])
	case "query":
		runQuery(os.Args[2:])
	case "graph":
		runGraph(os.Args[2:])
	case "tree":
		runTree(os.Args[2:])
	case "sql":
		runSQL(os.Args[2:])
	case "router":
		runRouter(os.Args[2:])
	case "mcp":
		runMCP(os.Args[2:])
	case "batch":
		runBatch(os.Args[2:])
	case "eval":
		runEval(os.Args[2:])
	default:
		usage()
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `uso:
  urag add -source <path> -db <path>
  urag query -q "pergunta" -db <path> [-k 5] [-where key=value]
  urag graph ask -source <path> -q "pergunta" [-llm-model <model>] [-hops 2] [-llm-provider ollama|openai] [-llm-base-url <url>] [-llm-api-key <key>] [-persist <dsn>]
  urag tree ask -source <path.md> -title <título> -q "pergunta" [-llm-model <model>] [-depth 3] [-llm-provider ollama|openai] [-llm-base-url <url>] [-llm-api-key <key>]
  urag sql query -dsn <path> -q "pergunta" [-llm-model <model>] [-llm-provider ollama|openai] [-llm-base-url <url>] [-llm-api-key <key>]
  urag router ask -source <path> -tree-source <path.md> -tree-title <título> -sql-dsn <path> -q "pergunta" [-k 5] [-llm-provider ollama|openai] [-llm-base-url <url>] [-llm-api-key <key>] [-graph-persist <dsn>]
  urag mcp serve [-db <path>] [-llm-model <model>] [-sql-dsn <path>] [-embedding-provider ollama|openai] [-embedding-model <model>] [-embedding-api-key <key>] [-embedding-base-url <url>] [-llm-provider ollama|openai] [-llm-base-url <url>] [-llm-api-key <key>] [-transport stdio|http] [-http-addr :8080] [-graph-persist <dsn>]
  urag graph stats -persist <dsn>
  urag graph export -persist <dsn> [-output <file>]
  urag batch submit -source <path> -db <path> [-workers 4]
  urag eval -q "pergunta" -answer "resposta" -context "contexto" [-ground-truth "resposta_ideal"] [-llm-provider ollama|openai] [-llm-model <model>] [-llm-base-url <url>] [-llm-api-key <key>]`)
}

func newRAG(dbPath string) (*rag.UnifiedRAG, error) {
	return rag.New(rag.Config{
		EmbeddingProvider: "ollama",
		EmbeddingModel:    "nomic-embed-text",
		PersistPath:       dbPath,
	})
}

func runAdd(args []string) {
	fs := flag.NewFlagSet("add", flag.ExitOnError)
	source := fs.String("source", "", "arquivo ou diretório de documentos")
	dbPath := fs.String("db", "./urag.db", "caminho do banco de persistência")
	fs.Parse(args)

	if *source == "" {
		fmt.Fprintln(os.Stderr, "erro: -source é obrigatório")
		os.Exit(1)
	}

	docs, err := loadDocuments(*source)
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao ler source:", err)
		os.Exit(1)
	}

	ur, err := newRAG(*dbPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao iniciar rag:", err)
		os.Exit(1)
	}

	if err := ur.AddDocuments(context.Background(), docs); err != nil {
		fmt.Fprintln(os.Stderr, "erro ao adicionar documentos:", err)
		os.Exit(1)
	}

	fmt.Printf("%d documento(s) adicionado(s) a %s\n", len(docs), *dbPath)
}

func runQuery(args []string) {
	fs := flag.NewFlagSet("query", flag.ExitOnError)
	question := fs.String("q", "", "pergunta")
	dbPath := fs.String("db", "./urag.db", "caminho do banco de persistência")
	topK := fs.Int("k", 5, "número de resultados")
	where := fs.String("where", "", "filtro de metadata, formato key=value,key2=value2")
	fs.Parse(args)

	if *question == "" {
		fmt.Fprintln(os.Stderr, "erro: -q é obrigatório")
		os.Exit(1)
	}

	whereMap, err := parseWhere(*where)
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro em -where:", err)
		os.Exit(1)
	}

	ur, err := newRAG(*dbPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao iniciar rag:", err)
		os.Exit(1)
	}

	results, err := ur.QueryFiltered(context.Background(), *question, *topK, whereMap, nil)
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro na query:", err)
		os.Exit(1)
	}

	for _, r := range results {
		fmt.Printf("[%.4f] %s: %s\n", r.Score, r.Document.ID, truncate(r.Document.Content, 120))
	}
}

// parseWhere converte "key=value,key2=value2" em map[string]string. Vazio retorna nil
// (sem filtro) — QueryFiltered com where=nil se comporta como Query sem filtro.
func parseWhere(s string) (map[string]string, error) {
	if s == "" {
		return nil, nil
	}
	out := map[string]string{}
	for _, pair := range strings.Split(s, ",") {
		k, v, ok := strings.Cut(pair, "=")
		if !ok {
			return nil, fmt.Errorf("par inválido %q, esperado key=value", pair)
		}
		out[k] = v
	}
	return out, nil
}

// loadDocuments lê um arquivo único, cada linha vira um Document.
// Diretórios (múltiplos arquivos) ficam para quando houver um segundo caso de uso real.
func loadDocuments(source string) ([]rag.Document, error) {
	data, err := os.ReadFile(source)
	if err != nil {
		return nil, err
	}

	base := filepath.Base(source)
	lines := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	docs := make([]rag.Document, 0, len(lines))
	for i, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		docs = append(docs, rag.Document{
			ID:      fmt.Sprintf("%s-%d", base, i),
			Content: line,
			Source:  base,
		})
	}
	return docs, nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

const defaultLLMModel = "granite4:micro-h"

func runGraph(args []string) {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "uso: urag graph <ask|stats|export> ...")
		os.Exit(1)
	}

	switch args[0] {
	case "ask":
		runGraphAsk(args[1:])
	case "stats":
		runGraphStats(args[1:])
	case "export":
		runGraphExport(args[1:])
	default:
		fmt.Fprintln(os.Stderr, "subcomando desconhecido:", args[0])
		os.Exit(1)
	}
}

func runGraphAsk(args []string) {
	fs := flag.NewFlagSet("graph ask", flag.ExitOnError)
	source := fs.String("source", "", "arquivo de documentos (uma linha = um documento)")
	question := fs.String("q", "", "pergunta")
	model := fs.String("llm-model", defaultLLMModel, "modelo para extração")
	hops := fs.Int("hops", 2, "distância máxima de navegação no grafo")
	provider := fs.String("llm-provider", "ollama", "provider do llm: ollama ou openai")
	baseURL := fs.String("llm-base-url", "", "override do endpoint (obrigatório pra openai-compatível que não é a OpenAI oficial)")
	apiKey := fs.String("llm-api-key", "", "API key, obrigatória se -llm-provider=openai")
	persistDSN := fs.String("persist", "", "DSN para persistência do grafo (ex: file:graph.db)")
	fs.Parse(args)

	if *source == "" || *question == "" {
		fmt.Fprintln(os.Stderr, "erro: -source e -q são obrigatórios")
		os.Exit(1)
	}

	docs, err := loadDocuments(*source)
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao ler source:", err)
		os.Exit(1)
	}

	// Configuração do Graph com persistência opcional
	graphConfig := graph.Config{
		LLMProvider:    *provider,
		LLMModel:       *model,
		LLMBaseURL:     *baseURL,
		LLMAPIKey:      *apiKey,
		PersistDSN:     *persistDSN,
		PersistEnabled: *persistDSN != "",
	}

	g, err := graph.New(graphConfig)
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao iniciar graph:", err)
		os.Exit(1)
	}
	defer g.Close()

	ctx := context.Background()
	if err := g.AddDocuments(ctx, docs); err != nil {
		fmt.Fprintln(os.Stderr, "erro ao extrair documentos:", err)
		os.Exit(1)
	}

	results, err := g.Query(ctx, *question, *hops)
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro na query:", err)
		os.Exit(1)
	}
	for _, rel := range results {
		fmt.Printf("%s --[%s]--> %s (doc: %s)\n", rel.Source, rel.Relation, rel.Target, rel.DocID)
	}

	if *persistDSN != "" {
		fmt.Printf("\n✅ Grafo persistido em: %s\n", *persistDSN)
	}
}

func runGraphStats(args []string) {
	fs := flag.NewFlagSet("graph stats", flag.ExitOnError)
	persistDSN := fs.String("persist", "", "DSN para persistência do grafo (ex: file:graph.db)")
	fs.Parse(args)

	if *persistDSN == "" {
		fmt.Fprintln(os.Stderr, "erro: -persist é obrigatório")
		os.Exit(1)
	}

	g, err := graph.New(graph.Config{
		PersistDSN:     *persistDSN,
		PersistEnabled: true,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao iniciar graph:", err)
		os.Exit(1)
	}
	defer g.Close()

	stats := g.GetStats()
	fmt.Printf("Estatísticas do Grafo (%s):\n", *persistDSN)
	fmt.Printf("  Entidades: %v\n", stats["entities"])
	fmt.Printf("  Relações:  %v\n", stats["relations"])
	fmt.Printf("  Persistido: %v\n", stats["persisted"])
}

func runGraphExport(args []string) {
	fs := flag.NewFlagSet("graph export", flag.ExitOnError)
	persistDSN := fs.String("persist", "", "DSN para persistência do grafo (ex: file:graph.db)")
	outputFile := fs.String("output", "", "arquivo de destino para o backup JSON (vazio = stdout)")
	fs.Parse(args)

	if *persistDSN == "" {
		fmt.Fprintln(os.Stderr, "erro: -persist é obrigatório")
		os.Exit(1)
	}

	g, err := graph.New(graph.Config{
		PersistDSN:     *persistDSN,
		PersistEnabled: true,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao iniciar graph:", err)
		os.Exit(1)
	}
	defer g.Close()

	snapshot, err := g.LoadFullGraph(context.Background())
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao carregar grafo:", err)
		os.Exit(1)
	}

	data, err := json.MarshalIndent(snapshot, "", "  ")
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao serializar grafo:", err)
		os.Exit(1)
	}

	if *outputFile != "" {
		if err := os.WriteFile(*outputFile, data, 0644); err != nil {
			fmt.Fprintln(os.Stderr, "erro ao gravar arquivo de saída:", err)
			os.Exit(1)
		}
		fmt.Printf("Grafo exportado com sucesso para %s\n", *outputFile)
	} else {
		fmt.Println(string(data))
	}
}

func runTree(args []string) {
	if len(args) < 1 || args[0] != "ask" {
		fmt.Fprintln(os.Stderr, "uso: urag tree ask -source <path.md> -title <título> -q \"pergunta\" [-llm-model <model>] [-depth 3]")
		os.Exit(1)
	}
	fs := flag.NewFlagSet("tree ask", flag.ExitOnError)
	source := fs.String("source", "", "arquivo markdown")
	title := fs.String("title", "", "título do documento")
	question := fs.String("q", "", "pergunta")
	model := fs.String("llm-model", defaultLLMModel, "modelo para navegação")
	depth := fs.Int("depth", 3, "profundidade máxima de navegação")
	provider := fs.String("llm-provider", "ollama", "provider do llm: ollama ou openai")
	baseURL := fs.String("llm-base-url", "", "override do endpoint (obrigatório pra openai-compatível que não é a OpenAI oficial)")
	apiKey := fs.String("llm-api-key", "", "API key, obrigatória se -llm-provider=openai")
	fs.Parse(args[1:])

	if *source == "" || *question == "" {
		fmt.Fprintln(os.Stderr, "erro: -source e -q são obrigatórios")
		os.Exit(1)
	}

	raw, err := os.ReadFile(*source)
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao ler source:", err)
		os.Exit(1)
	}

	docTitle := *title
	if docTitle == "" {
		docTitle = filepath.Base(*source)
	}

	t, err := tree.New(tree.Config{LLMProvider: *provider, LLMModel: *model, LLMBaseURL: *baseURL, LLMAPIKey: *apiKey})
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao iniciar tree:", err)
		os.Exit(1)
	}

	ctx := context.Background()
	if err := t.AddDocument(ctx, filepath.Base(*source), docTitle, string(raw)); err != nil {
		fmt.Fprintln(os.Stderr, "erro ao processar documento:", err)
		os.Exit(1)
	}

	results, err := t.Query(ctx, *question, *depth)
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro na query:", err)
		os.Exit(1)
	}
	for _, n := range results {
		fmt.Printf("%s: %s\n", n.Title, n.Content)
	}
}

func runSQL(args []string) {
	if len(args) < 1 || args[0] != "query" {
		fmt.Fprintln(os.Stderr, "uso: urag sql query -dsn <path> -q \"pergunta\" [-llm-model <model>]")
		os.Exit(1)
	}
	fs := flag.NewFlagSet("sql query", flag.ExitOnError)
	dsn := fs.String("dsn", "", "caminho do banco SQLite")
	question := fs.String("q", "", "pergunta")
	model := fs.String("llm-model", defaultLLMModel, "modelo para geração de SQL")
	provider := fs.String("llm-provider", "ollama", "provider do llm: ollama ou openai")
	baseURL := fs.String("llm-base-url", "", "override do endpoint (obrigatório pra openai-compatível que não é a OpenAI oficial)")
	apiKey := fs.String("llm-api-key", "", "API key, obrigatória se -llm-provider=openai")
	fs.Parse(args[1:])

	if *dsn == "" || *question == "" {
		fmt.Fprintln(os.Stderr, "erro: -dsn e -q são obrigatórios")
		os.Exit(1)
	}

	s, err := urasql.New(urasql.Config{DSN: *dsn, LLMProvider: *provider, LLMModel: *model, LLMBaseURL: *baseURL, LLMAPIKey: *apiKey})
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao iniciar sql:", err)
		os.Exit(1)
	}

	rows, generatedSQL, err := s.Query(context.Background(), *question)
	fmt.Println("SQL:", generatedSQL)
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro na query:", err)
		os.Exit(1)
	}
	for _, row := range rows {
		fmt.Printf("%+v\n", row)
	}
}

func runRouter(args []string) {
	if len(args) < 1 || args[0] != "ask" {
		fmt.Fprintln(os.Stderr, "uso: urag router ask -source <path> -tree-source <path.md> -tree-title <título> -sql-dsn <path> -q \"pergunta\" [-k 5] [-graph-persist <dsn>]")
		os.Exit(1)
	}
	fs := flag.NewFlagSet("router ask", flag.ExitOnError)
	source := fs.String("source", "", "arquivo de documentos para vector+graph (uma linha = um documento)")
	treeSource := fs.String("tree-source", "", "arquivo markdown para tree")
	treeTitle := fs.String("tree-title", "", "título do documento markdown")
	sqlDSN := fs.String("sql-dsn", "", "caminho do banco SQLite")
	question := fs.String("q", "", "pergunta")
	topK := fs.Int("k", 5, "número de resultados (vector)")
	model := fs.String("llm-model", defaultLLMModel, "modelo para classificação/extração/navegação/geração")
	provider := fs.String("llm-provider", "ollama", "provider do llm para graph/tree/sql: ollama ou openai (classificação do router continua ollama)")
	baseURL := fs.String("llm-base-url", "", "override do endpoint (obrigatório pra openai-compatível que não é a OpenAI oficial)")
	apiKey := fs.String("llm-api-key", "", "API key, obrigatória se -llm-provider=openai")
	graphPersist := fs.String("graph-persist", "", "DSN para persistência do grafo (ex: file:graph.db)")
	fs.Parse(args[1:])

	if *source == "" || *treeSource == "" || *sqlDSN == "" || *question == "" {
		fmt.Fprintln(os.Stderr, "erro: -source, -tree-source, -sql-dsn e -q são obrigatórios (Router exige as 4 stores configuradas)")
		os.Exit(1)
	}

	docs, err := loadDocuments(*source)
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao ler source:", err)
		os.Exit(1)
	}
	treeRaw, err := os.ReadFile(*treeSource)
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao ler tree-source:", err)
		os.Exit(1)
	}
	docTitle := *treeTitle
	if docTitle == "" {
		docTitle = filepath.Base(*treeSource)
	}

	r, err := router.New(router.Config{
		Vector: rag.Config{EmbeddingProvider: "ollama", EmbeddingModel: "nomic-embed-text"},
		Graph: graph.Config{
			LLMProvider:    *provider,
			LLMModel:       *model,
			LLMBaseURL:     *baseURL,
			LLMAPIKey:      *apiKey,
			PersistDSN:     *graphPersist,
			PersistEnabled: *graphPersist != "",
		},
		Tree:        tree.Config{LLMProvider: *provider, LLMModel: *model, LLMBaseURL: *baseURL, LLMAPIKey: *apiKey},
		SQL:         urasql.Config{DSN: *sqlDSN, LLMProvider: *provider, LLMModel: *model, LLMBaseURL: *baseURL, LLMAPIKey: *apiKey},
		RouterModel: *model,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao iniciar router:", err)
		os.Exit(1)
	}

	ctx := context.Background()
	if err := r.AddDocuments(ctx, docs); err != nil {
		fmt.Fprintln(os.Stderr, "erro ao adicionar documentos:", err)
		os.Exit(1)
	}
	if err := r.AddMarkdownDocument(ctx, filepath.Base(*treeSource), docTitle, string(treeRaw)); err != nil {
		fmt.Fprintln(os.Stderr, "erro ao processar documento markdown:", err)
		os.Exit(1)
	}

	result, err := r.Query(ctx, *question, *topK)
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro na query:", err)
		os.Exit(1)
	}

	fmt.Println("strategy:", result.Strategy)
	for _, sr := range result.Vector {
		fmt.Printf("  [vector %.4f] %s: %s\n", sr.Score, sr.Document.ID, truncate(sr.Document.Content, 120))
	}
	for _, rel := range result.Graph {
		fmt.Printf("  [graph] %s --[%s]--> %s\n", rel.Source, rel.Relation, rel.Target)
	}
	for _, n := range result.Tree {
		fmt.Printf("  [tree] %s: %s\n", n.Title, n.Content)
	}
	if result.SQLQuery != "" {
		fmt.Printf("  [sql] %s -> %+v\n", result.SQLQuery, result.SQLRows)
	}
}

func runMCP(args []string) {
	if len(args) < 1 || args[0] != "serve" {
		fmt.Fprintln(os.Stderr, "uso: urag mcp serve [-db <path>] [-llm-model <model>] [-sql-dsn <path>] [-embedding-provider ollama|openai] [-embedding-model <model>] [-embedding-api-key <key>] [-graph-persist <dsn>] [-embedding-cache-size <size>] [-embedding-cache-ttl <ttl>]")
		os.Exit(1)
	}
	fs := flag.NewFlagSet("mcp serve", flag.ExitOnError)
	dbPath := fs.String("db", getEnv("URAG_DB", "./urag_mcp.db"), `caminho do banco de persistência do vector store ("" = in-memory)`)
	model := fs.String("llm-model", getEnv("URAG_LLM_MODEL", defaultLLMModel), "modelo para graph/tree/sql")
	sqlDSN := fs.String("sql-dsn", getEnv("URAG_SQL_DSN", ""), "caminho do banco SQLite (vazio = urag_sql.db por padrão)")
	embeddingProvider := fs.String("embedding-provider", getEnv("URAG_EMBEDDING_PROVIDER", "ollama"), "provider de embedding do vector store: ollama ou openai")
	embeddingModel := fs.String("embedding-model", getEnv("URAG_EMBEDDING_MODEL", ""), "modelo de embedding (default: nomic-embed-text pra ollama)")
	embeddingAPIKey := fs.String("embedding-api-key", getEnv("URAG_EMBEDDING_API_KEY", ""), "API key, obrigatória se -embedding-provider=openai")
	embeddingBaseURL := fs.String("embedding-base-url", getEnv("URAG_EMBEDDING_BASE_URL", ""), "override do endpoint Ollama de embedding (ex: http://ollama:11434 num docker-compose)")
	llmProvider := fs.String("llm-provider", getEnv("URAG_LLM_PROVIDER", "ollama"), "provider do llm para graph/tree/sql: ollama ou openai")
	llmBaseURL := fs.String("llm-base-url", getEnv("URAG_LLM_BASE_URL", ""), "override do endpoint (obrigatório pra openai-compatível que não é a OpenAI oficial)")
	llmAPIKey := fs.String("llm-api-key", getEnv("URAG_LLM_API_KEY", ""), "API key, obrigatória se -llm-provider=openai")
	transport := fs.String("transport", getEnv("URAG_TRANSPORT", "stdio"), "transporte MCP: stdio (default, subprocesso local) ou http (Streamable HTTP/SSE, pra UI web ou clientes na rede)")
	httpAddr := fs.String("http-addr", getEnv("URAG_HTTP_ADDR", ":8080"), "endereço de escuta quando -transport=http")
	graphPersist := fs.String("graph-persist", getEnv("URAG_GRAPH_PERSIST", ""), "DSN para persistência do grafo (ex: file:graph.db)")
	metricsPort := fs.String("metrics-port", getEnv("URAG_METRICS_PORT", ""), "porta exclusiva para expor métricas do Prometheus (ex: :9090)")

	// Configuração do cache de embeddings via flags / env vars
	cacheSizeStr := getEnv("URAG_EMBEDDING_CACHE_SIZE", "1000")
	var defaultCacheSize int
	fmt.Sscanf(cacheSizeStr, "%d", &defaultCacheSize)

	cacheTTLStr := getEnv("URAG_EMBEDDING_CACHE_TTL", "5m")
	defaultCacheTTL, err := time.ParseDuration(cacheTTLStr)
	if err != nil {
		defaultCacheTTL = 5 * time.Minute
	}

	embeddingCacheSize := fs.Int("embedding-cache-size", defaultCacheSize, "tamanho máximo do cache de embeddings")
	embeddingCacheTTL := fs.Duration("embedding-cache-ttl", defaultCacheTTL, "TTL dos itens do cache de embeddings (ex: 5m, 1h)")

	fs.Parse(args[1:])

	s, err := mcpserver.New(mcpserver.Config{
		VectorDBPath:       *dbPath,
		EmbeddingProvider:  *embeddingProvider,
		EmbeddingModel:     *embeddingModel,
		EmbeddingAPIKey:    *embeddingAPIKey,
		EmbeddingBaseURL:   *embeddingBaseURL,
		LLMProvider:        *llmProvider,
		LLMModel:           *model,
		LLMBaseURL:         *llmBaseURL,
		LLMAPIKey:          *llmAPIKey,
		SQLDSN:             *sqlDSN,
		GraphPersist:       *graphPersist,
		EmbeddingCacheSize: *embeddingCacheSize,
		EmbeddingCacheTTL:  *embeddingCacheTTL,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao iniciar servidor mcp:", err)
		os.Exit(1)
	}
	defer s.Close()

	if *metricsPort != "" {
		port := *metricsPort
		if !strings.HasPrefix(port, ":") {
			port = ":" + port
		}
		// Sobe servidor HTTP de métricas dedicado em background
		go func() {
			mux := http.NewServeMux()
			mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
				w.Header().Set("Cache-Control", "no-cache")
				w.WriteHeader(http.StatusOK)
				w.Write([]byte(telemetry.GlobalCollector.RenderPrometheus()))
			})
			fmt.Fprintf(os.Stderr, "📊 Servidor de métricas Prometheus ativo em http://localhost%s/metrics\n", port)
			if err := http.ListenAndServe(port, mux); err != nil {
				fmt.Fprintf(os.Stderr, "erro no servidor de métricas: %v\n", err)
			}
		}()
	}

	switch *transport {
	case "stdio":
		err = s.Run(context.Background())
	case "http":
		fmt.Fprintln(os.Stderr, "servidor mcp (http) escutando em", *httpAddr)
		err = s.RunHTTP(context.Background(), *httpAddr)
	default:
		fmt.Fprintf(os.Stderr, "erro: -transport desconhecido: %q (use stdio ou http)\n", *transport)
		os.Exit(1)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro no servidor mcp:", err)
		os.Exit(1)
	}
}

// ==================== UTILITY FUNCTIONS ====================

// loadEnv loads environment variables from .env file
func loadEnv() {
	file, err := os.Open(".env")
	if err != nil {
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(parts[1])
		if len(val) >= 2 && ((val[0] == '"' && val[len(val)-1] == '"') || (val[0] == '\'' && val[len(val)-1] == '\'')) {
			val = val[1 : len(val)-1]
		}
		if os.Getenv(key) == "" {
			os.Setenv(key, val)
		}
	}
}

// getEnv returns environment variable or default value
func getEnv(key, defaultValue string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultValue
}

func runBatch(args []string) {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "uso: urag batch [submit] ...")
		os.Exit(1)
	}

	switch args[0] {
	case "submit":
		runBatchSubmit(args[1:])
	default:
		fmt.Fprintln(os.Stderr, "subcomando desconhecido:", args[0])
		os.Exit(1)
	}
}

func runBatchSubmit(args []string) {
	fs := flag.NewFlagSet("batch submit", flag.ExitOnError)
	source := fs.String("source", "", "arquivo ou diretório de documentos")
	dbPath := fs.String("db", "./urag.db", "caminho do banco de persistência")
	workers := fs.Int("workers", 4, "número de workers concorrentes")
	fs.Parse(args)

	if *source == "" {
		fmt.Fprintln(os.Stderr, "erro: -source é obrigatório")
		os.Exit(1)
	}

	docs, err := loadDocuments(*source)
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao ler source:", err)
		os.Exit(1)
	}

	ur, err := newRAG(*dbPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao iniciar RAG:", err)
		os.Exit(1)
	}
	defer ur.Close()

	bp := rag.NewBatchProcessor(*workers, len(docs))
	ctx := context.Background()
	bp.Start(ctx, ur)

	fmt.Printf("Submetendo lote de %d documentos com %d workers...\n", len(docs), *workers)
	jobID, err := bp.Submit(docs)
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao submeter lote:", err)
		os.Exit(1)
	}

	// Loop bloqueante de progresso para a CLI
	start := time.Now()
	for {
		status, err := bp.GetStatus(jobID)
		if err != nil {
			fmt.Fprintln(os.Stderr, "erro ao obter status do lote:", err)
			os.Exit(1)
		}

		fmt.Printf("\rProgresso: %d/%d concluídos (falhas: %d, status: %s)...", status.Completed, status.Total, status.Failed, status.Status)

		if status.Status == "completed" || status.Status == "failed" {
			fmt.Println()
			fmt.Printf("\n✨ Lote finalizado em %v!\n", time.Since(start).Truncate(time.Millisecond))
			if status.Failed > 0 {
				fmt.Printf("⚠️  Houve %d falhas. Lista de erros:\n", status.Failed)
				for _, errMsg := range status.Errors {
					fmt.Printf("  - %s\n", errMsg)
				}
			} else {
				fmt.Println("✅ Todos os documentos foram indexados com sucesso!")
			}
			break
		}

		time.Sleep(200 * time.Millisecond)
	}
}

func runEval(args []string) {
	fs := flag.NewFlagSet("eval", flag.ExitOnError)
	question := fs.String("q", "", "pergunta do usuário")
	answer := fs.String("answer", "", "resposta gerada pelo RAG")
	contextText := fs.String("context", "", "contexto recuperado do RAG")
	groundTruth := fs.String("ground-truth", "", "resposta ideal (opcional)")
	provider := fs.String("llm-provider", "ollama", "provider do llm: ollama ou openai")
	model := fs.String("llm-model", defaultLLMModel, "modelo LLM para avaliação")
	baseURL := fs.String("llm-base-url", "", "override do endpoint LLM")
	apiKey := fs.String("llm-api-key", "", "API key do provider")
	embedProvider := fs.String("embedding-provider", "ollama", "provider de embedding: ollama ou openai")
	embedModel := fs.String("embedding-model", "", "modelo de embedding (default: nomic-embed-text pra ollama)")
	fs.Parse(args)

	if *question == "" || *answer == "" || *contextText == "" {
		fmt.Fprintln(os.Stderr, "erro: -q, -answer e -context são obrigatórios")
		os.Exit(1)
	}

	embModel := *embedModel
	if embModel == "" {
		if *embedProvider == "openai" {
			embModel = "text-embedding-3-small"
		} else {
			embModel = "nomic-embed-text"
		}
	}

	// Inicializa UnifiedRAG temporário em memória para geração de embeddings
	ur, err := rag.New(rag.Config{
		EmbeddingProvider: *embedProvider,
		EmbeddingModel:    embModel,
		EmbeddingAPIKey:   *apiKey,
		EmbeddingBaseURL:  *baseURL,
		PersistPath:       "", // in-memory
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao iniciar gerador de embeddings:", err)
		os.Exit(1)
	}
	defer ur.Close()

	embedFunc := func(ctx context.Context, text string) ([]float32, error) {
		return ur.GenerateEmbedding(ctx, text)
	}

	evaluator := eval.NewEvaluator(*provider, *model, *baseURL, *apiKey)

	fmt.Println("Calculando métricas RAGAS...")
	metrics, err := evaluator.Evaluate(context.Background(), *question, *answer, *contextText, *groundTruth, embedFunc)
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro na avaliação:", err)
		os.Exit(1)
	}

	fmt.Println("\n====================================")
	fmt.Println("📊 RESULTADOS DA AVALIAÇÃO RAGAS")
	fmt.Println("====================================")
	fmt.Printf("  Fidelidade (Faithfulness):     %.2f\n", metrics.Faithfulness)
	fmt.Printf("  Relevância (Answer Relevancy): %.2f\n", metrics.AnswerRelevancy)
	if *groundTruth != "" {
		fmt.Printf("  Recall (Context Recall):       %.2f\n", metrics.ContextRecall)
	} else {
		fmt.Println("  Recall (Context Recall):       [Ignorado - ground-truth não fornecido]")
	}
	fmt.Println("====================================")
}
