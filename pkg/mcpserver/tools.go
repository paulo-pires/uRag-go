package mcpserver

import (
	"context"
	"fmt"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"urag-go/pkg/rag"
)

// registerTools registra as tools MCP: 6 sempre (vector/graph/tree, add+query
// cada), mais sql_query/sql_load só se s.sql != nil.
func (s *Server) registerTools() {
	mcp.AddTool(s.mcp, &mcp.Tool{Name: "vector_add", Description: "Adiciona documentos ao Vector RAG (busca por similaridade semântica)."}, s.vectorAdd)
	mcp.AddTool(s.mcp, &mcp.Tool{Name: "vector_query", Description: "Busca por similaridade semântica no Vector RAG. Retorna confidence score em cada resultado."}, s.vectorQuery)
	mcp.AddTool(s.mcp, &mcp.Tool{Name: "graph_add", Description: "Extrai entidades/relações dos documentos e adiciona ao Graph RAG (persistente se configurado)."}, s.graphAdd)
	mcp.AddTool(s.mcp, &mcp.Tool{Name: "graph_query", Description: "Busca multi-hop sobre o grafo de entidades/relações."}, s.graphQuery)
	mcp.AddTool(s.mcp, &mcp.Tool{Name: "graph_stats", Description: "Retorna estatísticas do grafo (número de entidades e relações)."}, s.graphStats)
	mcp.AddTool(s.mcp, &mcp.Tool{Name: "tree_add", Description: "Adiciona um documento markdown ao Vectorless RAG, parseado em árvore de headings."}, s.treeAdd)
	mcp.AddTool(s.mcp, &mcp.Tool{Name: "tree_query", Description: "Navega a árvore hierárquica do documento para responder à pergunta."}, s.treeQuery)
	mcp.AddTool(s.mcp, &mcp.Tool{
		Name:        "paper_ingest",
		Description: "Ingere um artigo científico estruturado (DOI, autores, ano, validated). Indexa no Vector RAG e extrai relações de citação para o Graph RAG. Use where:{\"validated\":\"true\"} no vector_query para filtrar só fontes validadas por pares.",
	}, s.paperIngest)
	if s.router != nil {
		mcp.AddTool(s.mcp, &mcp.Tool{Name: "router_query", Description: "Roteamento automático: detecta qual store (vector/graph/tree/sql) melhor responde a pergunta e executa a busca."}, s.routerQuery)
	}
	if s.sql != nil {
		mcp.AddTool(s.mcp, &mcp.Tool{Name: "sql_query", Description: "Converte a pergunta em SQL (SELECT) e executa contra o banco configurado."}, s.sqlQuery)
		mcp.AddTool(s.mcp, &mcp.Tool{Name: "sql_load", Description: "Cria uma tabela e importa dados estruturados em CSV ou JSON."}, s.sqlLoad)
	}
}

// DocumentInput espelha rag.Document nos campos que a tool aceita.
type DocumentInput struct {
	ID      string            `json:"id" jsonschema:"identificador único do documento"`
	Content string            `json:"content" jsonschema:"conteúdo do documento"`
	Source  string            `json:"source,omitempty" jsonschema:"origem do documento"`
	Meta    map[string]string `json:"meta,omitempty" jsonschema:"metadata para filtros (where)"`
}

func (d DocumentInput) toDocument() rag.Document {
	return rag.Document{ID: d.ID, Content: d.Content, Source: d.Source, Meta: d.Meta}
}

type VectorAddInput struct {
	Documents []DocumentInput `json:"documents" jsonschema:"documentos a adicionar ao Vector RAG"`
}
type VectorAddOutput struct {
	Added int `json:"added"`
}

func (s *Server) vectorAdd(ctx context.Context, _ *mcp.CallToolRequest, in VectorAddInput) (*mcp.CallToolResult, VectorAddOutput, error) {
	docs := make([]rag.Document, len(in.Documents))
	for i, d := range in.Documents {
		docs[i] = d.toDocument()
	}
	if err := s.vector.AddDocuments(ctx, docs); err != nil {
		return nil, VectorAddOutput{}, err
	}
	return nil, VectorAddOutput{Added: len(docs)}, nil
}

type VectorQueryInput struct {
	Question string            `json:"question" jsonschema:"pergunta em linguagem natural"`
	TopK     int               `json:"top_k,omitempty" jsonschema:"número de resultados (default 5)"`
	Where    map[string]string `json:"where,omitempty" jsonschema:"filtro de metadata (exact match)"`
}
type SearchResultOutput struct {
	Document   DocumentInput `json:"document"`
	Score      float32       `json:"score"`
	Confidence float64       `json:"confidence"` // 0.0-1.0, normalizado para uso pelo agente
}
type VectorQueryOutput struct {
	Results    []SearchResultOutput `json:"results"`
	Confidence float64              `json:"confidence"` // confiança média dos top resultados
}

func (s *Server) vectorQuery(ctx context.Context, _ *mcp.CallToolRequest, in VectorQueryInput) (*mcp.CallToolResult, VectorQueryOutput, error) {
	topK := in.TopK
	if topK <= 0 {
		topK = 5
	}
	results, err := s.vector.QueryFiltered(ctx, in.Question, topK, in.Where, nil)
	if err != nil {
		return nil, VectorQueryOutput{}, err
	}
	out := make([]SearchResultOutput, len(results))
	var totalConfidence float64
	for i, r := range results {
		// Normaliza score coseno (tipicamente 0.0-1.0) para confidence
		conf := float64(r.Score)
		if conf < 0 {
			conf = 0
		}
		if conf > 1 {
			conf = 1
		}
		totalConfidence += conf
		out[i] = SearchResultOutput{
			Document:   DocumentInput{ID: r.Document.ID, Content: r.Document.Content, Source: r.Document.Source, Meta: r.Document.Meta},
			Score:      r.Score,
			Confidence: conf,
		}
	}
	avgConf := 0.0
	if len(results) > 0 {
		avgConf = totalConfidence / float64(len(results))
	}
	return nil, VectorQueryOutput{Results: out, Confidence: avgConf}, nil
}

type GraphAddInput struct {
	Documents []DocumentInput `json:"documents" jsonschema:"documentos dos quais extrair entidades/relações"`
}
type GraphAddOutput struct {
	Added     int    `json:"added"`
	Persisted bool   `json:"persisted,omitempty"`
	DSN       string `json:"dsn,omitempty"`
}

func (s *Server) graphAdd(ctx context.Context, _ *mcp.CallToolRequest, in GraphAddInput) (*mcp.CallToolResult, GraphAddOutput, error) {
	docs := make([]rag.Document, len(in.Documents))
	for i, d := range in.Documents {
		docs[i] = d.toDocument()
	}
	if err := s.graph.AddDocuments(ctx, docs); err != nil {
		return nil, GraphAddOutput{}, err
	}

	output := GraphAddOutput{
		Added: len(docs),
	}

	// Verifica se a persistência está habilitada
	if s.config.GraphPersist != "" {
		output.Persisted = true
		output.DSN = s.config.GraphPersist
	}

	return nil, output, nil
}

type GraphQueryInput struct {
	Question string `json:"question" jsonschema:"pergunta em linguagem natural"`
	Hops     int    `json:"hops,omitempty" jsonschema:"distância máxima de navegação no grafo (default 2)"`
}
type RelationOutput struct {
	Source   string `json:"source"`
	Target   string `json:"target"`
	Relation string `json:"relation"`
	DocID    string `json:"doc_id"`
}
type GraphQueryOutput struct {
	Relations  []RelationOutput       `json:"relations"`
	Stats      map[string]interface{} `json:"stats,omitempty"`
	Persistent bool                   `json:"persistent,omitempty"`
}

func (s *Server) graphQuery(ctx context.Context, _ *mcp.CallToolRequest, in GraphQueryInput) (*mcp.CallToolResult, GraphQueryOutput, error) {
	hops := in.Hops
	if hops <= 0 {
		hops = 2
	}
	rels, err := s.graph.Query(ctx, in.Question, hops)
	if err != nil {
		return nil, GraphQueryOutput{}, err
	}

	out := make([]RelationOutput, len(rels))
	for i, r := range rels {
		out[i] = RelationOutput{Source: r.Source, Target: r.Target, Relation: r.Relation, DocID: r.DocID}
	}

	output := GraphQueryOutput{
		Relations: out,
		Stats:     s.graph.GetStats(),
	}

	if s.config.GraphPersist != "" {
		output.Persistent = true
	}

	return nil, output, nil
}

// graphStats retorna estatísticas do grafo
func (s *Server) graphStats(ctx context.Context, _ *mcp.CallToolRequest, _ struct{}) (*mcp.CallToolResult, map[string]interface{}, error) {
	stats := s.graph.GetStats()

	if s.config.GraphPersist != "" {
		stats["persistent"] = true
		stats["dsn"] = s.config.GraphPersist
	}

	return nil, stats, nil
}

type TreeAddInput struct {
	ID       string `json:"id" jsonschema:"identificador único do documento"`
	Title    string `json:"title" jsonschema:"título do documento"`
	Markdown string `json:"markdown" jsonschema:"conteúdo markdown do documento (headings viram a árvore)"`
}
type TreeAddOutput struct {
	OK bool `json:"ok"`
}

func (s *Server) treeAdd(ctx context.Context, _ *mcp.CallToolRequest, in TreeAddInput) (*mcp.CallToolResult, TreeAddOutput, error) {
	if err := s.tree.AddDocument(ctx, in.ID, in.Title, in.Markdown); err != nil {
		return nil, TreeAddOutput{}, err
	}
	return nil, TreeAddOutput{OK: true}, nil
}

type TreeQueryInput struct {
	Question string `json:"question" jsonschema:"pergunta em linguagem natural"`
	MaxDepth int    `json:"max_depth,omitempty" jsonschema:"profundidade máxima de navegação (default 3)"`
}
type NodeOutput struct {
	Title   string `json:"title"`
	Content string `json:"content"`
}
type TreeQueryOutput struct {
	Nodes []NodeOutput `json:"nodes"`
}

func (s *Server) treeQuery(ctx context.Context, _ *mcp.CallToolRequest, in TreeQueryInput) (*mcp.CallToolResult, TreeQueryOutput, error) {
	depth := in.MaxDepth
	if depth <= 0 {
		depth = 3
	}
	nodes, err := s.tree.Query(ctx, in.Question, depth)
	if err != nil {
		return nil, TreeQueryOutput{}, err
	}
	out := make([]NodeOutput, len(nodes))
	for i, n := range nodes {
		out[i] = NodeOutput{Title: n.Title, Content: n.Content}
	}
	return nil, TreeQueryOutput{Nodes: out}, nil
}

type SQLQueryInput struct {
	Question string `json:"question" jsonschema:"pergunta em linguagem natural"`
}
type SQLQueryOutput struct {
	SQL  string           `json:"sql"`
	Rows []map[string]any `json:"rows"`
}

func (s *Server) sqlQuery(ctx context.Context, _ *mcp.CallToolRequest, in SQLQueryInput) (*mcp.CallToolResult, SQLQueryOutput, error) {
	rows, generatedSQL, err := s.sql.Query(ctx, in.Question)
	if err != nil {
		return nil, SQLQueryOutput{SQL: generatedSQL}, err
	}
	return nil, SQLQueryOutput{SQL: generatedSQL, Rows: rows}, nil
}

type SQLLoadInput struct {
	TableName string `json:"table_name" jsonschema:"nome da tabela a ser criada e populada"`
	Format    string `json:"format" jsonschema:"formato dos dados: csv ou json"`
	Data      string `json:"data" jsonschema:"conteúdo textual dos dados (CSV com header ou JSON array/objeto)"`
}
type SQLLoadOutput struct {
	OK bool `json:"ok"`
}

func (s *Server) sqlLoad(ctx context.Context, _ *mcp.CallToolRequest, in SQLLoadInput) (*mcp.CallToolResult, SQLLoadOutput, error) {
	if err := s.sql.LoadData(ctx, in.TableName, in.Format, in.Data); err != nil {
		return nil, SQLLoadOutput{OK: false}, err
	}
	return nil, SQLLoadOutput{OK: true}, nil
}

// ─── Paper Ingest ─────────────────────────────────────────────────────────────

// PaperIngestInput descreve um artigo científico com metadados estruturados.
type PaperIngestInput struct {
	DOI        string   `json:"doi" jsonschema:"DOI do artigo (ex: 10.1021/...)"`
	Title      string   `json:"title" jsonschema:"título do artigo"`
	Authors    []string `json:"authors" jsonschema:"lista de autores"`
	Year       int      `json:"year" jsonschema:"ano de publicação"`
	Journal    string   `json:"journal,omitempty" jsonschema:"nome do periódico"`
	Abstract   string   `json:"abstract" jsonschema:"resumo do artigo"`
	Body       string   `json:"body,omitempty" jsonschema:"corpo completo do artigo (opcional)"`
	Validated  bool     `json:"validated" jsonschema:"true = revisado por pares / fonte confiável"`
	References []string `json:"references,omitempty" jsonschema:"lista de DOIs citados por este artigo"`
}

type PaperIngestOutput struct {
	ID         string `json:"id"`
	VectorOK   bool   `json:"vector_ok"`
	GraphOK    bool   `json:"graph_ok"`
	References int    `json:"references_indexed"`
}

func (s *Server) paperIngest(ctx context.Context, _ *mcp.CallToolRequest, in PaperIngestInput) (*mcp.CallToolResult, PaperIngestOutput, error) {
	if in.DOI == "" {
		return nil, PaperIngestOutput{}, fmt.Errorf("doi é obrigatório")
	}
	id := "paper:" + in.DOI

	// Conteúdo para embedding = título + abstract + body (quando presente)
	content := in.Title + "\n\n" + in.Abstract
	if in.Body != "" {
		content += "\n\n" + in.Body
	}

	validated := "false"
	if in.Validated {
		validated = "true"
	}
	meta := map[string]string{
		"doi":       in.DOI,
		"title":     in.Title,
		"year":      fmt.Sprintf("%d", in.Year),
		"journal":   in.Journal,
		"validated": validated,
		"authors":   joinStrings(in.Authors, "; "),
		"type":      "paper",
	}

	out := PaperIngestOutput{ID: id}

	// ── Vector RAG ────────────────────────────────────────────────────────────
	if err := s.vector.AddDocuments(ctx, []rag.Document{
		{ID: id, Content: content, Source: in.DOI, Meta: meta},
	}); err != nil {
		return nil, out, fmt.Errorf("vector_add: %w", err)
	}
	out.VectorOK = true

	// ── Graph RAG: relações de citação ────────────────────────────────────────
	// Indexa só o paper principal; referências codificadas no conteúdo para o
	// extrator do grafo criar arestas CITES sem gerar nós fantasma sem metadados.
	citeContent := in.Title
	if len(in.References) > 0 {
		citeContent += "\ncites: " + joinStrings(in.References, "; ")
	}
	if err := s.graph.AddDocuments(ctx, []rag.Document{
		{ID: id, Content: citeContent, Source: in.DOI},
	}); err == nil {
		out.GraphOK = true
	}
	out.References = len(in.References)

	return nil, out, nil
}

func joinStrings(ss []string, sep string) string {
	result := ""
	for i, s := range ss {
		if i > 0 {
			result += sep
		}
		result += s
	}
	return result
}

// ─── Router Query ─────────────────────────────────────────────────────────────

type RouterQueryInput struct {
	Question string `json:"question" jsonschema:"pergunta em linguagem natural para roteamento automático"`
	TopK     int    `json:"top_k,omitempty" jsonschema:"número de resultados para vector (default 5)"`
}

// RouterQueryOutput unifica resultados de todas as stores com estratégia escolhida.
type RouterQueryOutput struct {
	Strategy   string               `json:"strategy"`             // qual store foi usada
	Confidence float64              `json:"confidence"`           // confiança da decisão de roteamento
	Results    []SearchResultOutput `json:"results,omitempty"`    // vector
	Relations  []RelationOutput     `json:"relations,omitempty"`  // graph
	Nodes      []NodeOutput         `json:"nodes,omitempty"`      // tree
	SQL        string               `json:"sql,omitempty"`        // sql gerado
	SQLRows    []map[string]any     `json:"sql_rows,omitempty"`   // resultados sql
}

func (s *Server) routerQuery(ctx context.Context, _ *mcp.CallToolRequest, in RouterQueryInput) (*mcp.CallToolResult, RouterQueryOutput, error) {
	if s.router == nil {
		return nil, RouterQueryOutput{}, fmt.Errorf("router não configurado")
	}
	topK := in.TopK
	if topK <= 0 {
		topK = 5
	}

	result, err := s.router.Query(ctx, in.Question, topK)
	if err != nil {
		return nil, RouterQueryOutput{}, err
	}

	out := RouterQueryOutput{
		Strategy: string(result.Strategy),
	}

	// Mapeia resultados por tipo de store
	for _, r := range result.Vector {
		conf := float64(r.Score)
		if conf < 0 {
			conf = 0
		}
		if conf > 1 {
			conf = 1
		}
		out.Results = append(out.Results, SearchResultOutput{
			Document:   DocumentInput{ID: r.Document.ID, Content: r.Document.Content, Source: r.Document.Source, Meta: r.Document.Meta},
			Score:      r.Score,
			Confidence: conf,
		})
		out.Confidence += conf
	}
	if len(result.Vector) > 0 {
		out.Confidence /= float64(len(result.Vector))
	}

	for _, rel := range result.Graph {
		out.Relations = append(out.Relations, RelationOutput{
			Source: rel.Source, Target: rel.Target, Relation: rel.Relation, DocID: rel.DocID,
		})
	}

	for _, n := range result.Tree {
		out.Nodes = append(out.Nodes, NodeOutput{Title: n.Title, Content: n.Content})
	}

	if result.SQLQuery != "" {
		out.SQL = result.SQLQuery
		out.SQLRows = result.SQLRows
	}

	return nil, out, nil
}
