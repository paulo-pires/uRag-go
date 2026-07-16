// Command urag é a CLI do uRag-go: add/query sobre Vector RAG (persistente),
// e comandos "ask" single-shot para graph/tree/router (in-memory, sem
// persistência — carregam e consultam na mesma invocação).
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"urag-go/pkg/graph"
	"urag-go/pkg/mcpserver"
	"urag-go/pkg/rag"
	"urag-go/pkg/router"
	urasql "urag-go/pkg/sql"
	"urag-go/pkg/tree"
)

func main() {
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
	default:
		usage()
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `uso:
  urag add -source <path> -db <path>
  urag query -q "pergunta" -db <path> [-k 5] [-where key=value]
  urag graph ask -source <path> -q "pergunta" [-llm-model <model>] [-hops 2] [-llm-provider ollama|openai] [-llm-base-url <url>] [-llm-api-key <key>]
  urag tree ask -source <path.md> -title <título> -q "pergunta" [-llm-model <model>] [-depth 3] [-llm-provider ollama|openai] [-llm-base-url <url>] [-llm-api-key <key>]
  urag sql query -dsn <path> -q "pergunta" [-llm-model <model>] [-llm-provider ollama|openai] [-llm-base-url <url>] [-llm-api-key <key>]
  urag router ask -source <path> -tree-source <path.md> -tree-title <título> -sql-dsn <path> -q "pergunta" [-k 5] [-llm-provider ollama|openai] [-llm-base-url <url>] [-llm-api-key <key>]
  urag mcp serve [-db <path>] [-llm-model <model>] [-sql-dsn <path>] [-embedding-provider ollama|openai] [-embedding-model <model>] [-embedding-api-key <key>] [-embedding-base-url <url>] [-llm-provider ollama|openai] [-llm-base-url <url>] [-llm-api-key <key>] [-transport stdio|http] [-http-addr :8080]`)
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
	if len(args) < 1 || args[0] != "ask" {
		fmt.Fprintln(os.Stderr, "uso: urag graph ask -source <path> -q \"pergunta\" [-llm-model <model>] [-hops 2]")
		os.Exit(1)
	}
	fs := flag.NewFlagSet("graph ask", flag.ExitOnError)
	source := fs.String("source", "", "arquivo de documentos (uma linha = um documento)")
	question := fs.String("q", "", "pergunta")
	model := fs.String("llm-model", defaultLLMModel, "modelo para extração")
	hops := fs.Int("hops", 2, "distância máxima de navegação no grafo")
	provider := fs.String("llm-provider", "ollama", "provider do llm: ollama ou openai")
	baseURL := fs.String("llm-base-url", "", "override do endpoint (obrigatório pra openai-compatível que não é a OpenAI oficial)")
	apiKey := fs.String("llm-api-key", "", "API key, obrigatória se -llm-provider=openai")
	fs.Parse(args[1:])

	if *source == "" || *question == "" {
		fmt.Fprintln(os.Stderr, "erro: -source e -q são obrigatórios")
		os.Exit(1)
	}

	docs, err := loadDocuments(*source)
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao ler source:", err)
		os.Exit(1)
	}

	g, err := graph.New(graph.Config{LLMProvider: *provider, LLMModel: *model, LLMBaseURL: *baseURL, LLMAPIKey: *apiKey})
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao iniciar graph:", err)
		os.Exit(1)
	}

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
		fmt.Fprintln(os.Stderr, "uso: urag router ask -source <path> -tree-source <path.md> -tree-title <título> -sql-dsn <path> -q \"pergunta\" [-k 5]")
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
		Vector:      rag.Config{EmbeddingProvider: "ollama", EmbeddingModel: "nomic-embed-text"},
		Graph:       graph.Config{LLMProvider: *provider, LLMModel: *model, LLMBaseURL: *baseURL, LLMAPIKey: *apiKey},
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
		fmt.Fprintln(os.Stderr, "uso: urag mcp serve [-db <path>] [-llm-model <model>] [-sql-dsn <path>] [-embedding-provider ollama|openai] [-embedding-model <model>] [-embedding-api-key <key>]")
		os.Exit(1)
	}
	fs := flag.NewFlagSet("mcp serve", flag.ExitOnError)
	dbPath := fs.String("db", "./urag_mcp.db", `caminho do banco de persistência do vector store ("" = in-memory)`)
	model := fs.String("llm-model", defaultLLMModel, "modelo para graph/tree/sql")
	sqlDSN := fs.String("sql-dsn", "", "caminho do banco SQLite (vazio = tool sql_query não é registrada)")
	embeddingProvider := fs.String("embedding-provider", "ollama", "provider de embedding do vector store: ollama ou openai")
	embeddingModel := fs.String("embedding-model", "", "modelo de embedding (default: nomic-embed-text pra ollama)")
	embeddingAPIKey := fs.String("embedding-api-key", "", "API key, obrigatória se -embedding-provider=openai")
	embeddingBaseURL := fs.String("embedding-base-url", "", "override do endpoint Ollama de embedding (ex: http://ollama:11434 num docker-compose)")
	llmProvider := fs.String("llm-provider", "ollama", "provider do llm para graph/tree/sql: ollama ou openai")
	llmBaseURL := fs.String("llm-base-url", "", "override do endpoint (obrigatório pra openai-compatível que não é a OpenAI oficial)")
	llmAPIKey := fs.String("llm-api-key", "", "API key, obrigatória se -llm-provider=openai")
	transport := fs.String("transport", "stdio", "transporte MCP: stdio (default, subprocesso local) ou http (Streamable HTTP/SSE, pra UI web ou clientes na rede)")
	httpAddr := fs.String("http-addr", ":8080", "endereço de escuta quando -transport=http")
	fs.Parse(args[1:])

	s, err := mcpserver.New(mcpserver.Config{
		VectorDBPath:      *dbPath,
		EmbeddingProvider: *embeddingProvider,
		EmbeddingModel:    *embeddingModel,
		EmbeddingAPIKey:   *embeddingAPIKey,
		EmbeddingBaseURL:  *embeddingBaseURL,
		LLMProvider:       *llmProvider,
		LLMModel:          *model,
		LLMBaseURL:        *llmBaseURL,
		LLMAPIKey:         *llmAPIKey,
		SQLDSN:            *sqlDSN,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "erro ao iniciar servidor mcp:", err)
		os.Exit(1)
	}
	defer s.Close()

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
