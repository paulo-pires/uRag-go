package mcpserver

import (
	"context"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"urag-go/pkg/rag"
)

// registerTools registra as tools MCP: 6 sempre (vector/graph/tree, add+query
// cada), mais sql_query/sql_load só se s.sql != nil.
func (s *Server) registerTools() {
	mcp.AddTool(s.mcp, &mcp.Tool{Name: "vector_add", Description: "Adiciona documentos ao Vector RAG (busca por similaridade semântica)."}, s.vectorAdd)
	mcp.AddTool(s.mcp, &mcp.Tool{Name: "vector_query", Description: "Busca por similaridade semântica no Vector RAG."}, s.vectorQuery)
	mcp.AddTool(s.mcp, &mcp.Tool{Name: "graph_add", Description: "Extrai entidades/relações dos documentos e adiciona ao Graph RAG."}, s.graphAdd)
	mcp.AddTool(s.mcp, &mcp.Tool{Name: "graph_query", Description: "Busca multi-hop sobre o grafo de entidades/relações."}, s.graphQuery)
	mcp.AddTool(s.mcp, &mcp.Tool{Name: "tree_add", Description: "Adiciona um documento markdown ao Vectorless RAG, parseado em árvore de headings."}, s.treeAdd)
	mcp.AddTool(s.mcp, &mcp.Tool{Name: "tree_query", Description: "Navega a árvore hierárquica do documento para responder à pergunta."}, s.treeQuery)
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
	Document DocumentInput `json:"document"`
	Score    float32       `json:"score"`
}
type VectorQueryOutput struct {
	Results []SearchResultOutput `json:"results"`
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
	for i, r := range results {
		out[i] = SearchResultOutput{
			Document: DocumentInput{ID: r.Document.ID, Content: r.Document.Content, Source: r.Document.Source, Meta: r.Document.Meta},
			Score:    r.Score,
		}
	}
	return nil, VectorQueryOutput{Results: out}, nil
}

type GraphAddInput struct {
	Documents []DocumentInput `json:"documents" jsonschema:"documentos dos quais extrair entidades/relações"`
}
type GraphAddOutput struct {
	Added int `json:"added"`
}

func (s *Server) graphAdd(ctx context.Context, _ *mcp.CallToolRequest, in GraphAddInput) (*mcp.CallToolResult, GraphAddOutput, error) {
	docs := make([]rag.Document, len(in.Documents))
	for i, d := range in.Documents {
		docs[i] = d.toDocument()
	}
	if err := s.graph.AddDocuments(ctx, docs); err != nil {
		return nil, GraphAddOutput{}, err
	}
	return nil, GraphAddOutput{Added: len(docs)}, nil
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
	Relations []RelationOutput `json:"relations"`
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
	return nil, GraphQueryOutput{Relations: out}, nil
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
