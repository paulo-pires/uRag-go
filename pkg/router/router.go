// Package router decide, por pergunta, qual store deve responder — orquestra
// pkg/rag, pkg/graph, pkg/tree e pkg/sql por cima, sem alterar nenhum deles.
package router

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"urag-go/internal/ollama"
	"urag-go/pkg/graph"
	"urag-go/pkg/rag"
	"urag-go/pkg/sql"
	"urag-go/pkg/tree"
)

// Strategy identifica qual store respondeu a uma Query.
type Strategy string

const (
	StrategyVector Strategy = "vector"
	StrategyGraph  Strategy = "graph"
	// StrategyBoth: pergunta que precisa de busca factual E relação entre
	// entidades — Query despacha pras duas stores em vez de escolher uma.
	StrategyBoth Strategy = "both"
	// StrategyTree: pergunta sobre um documento estruturado (Vectorless RAG).
	StrategyTree Strategy = "tree"
	// StrategySQL: pergunta sobre dados tabulares (Text-to-SQL).
	StrategySQL Strategy = "sql"

	// defaultRouterModel: granite4:micro-h. Testado contra qwen3.5:0.8b (modelo de
	// "thinking" — a resposta final vai pro campo thinking, não pro campo response
	// que ollama.Complete lê, então a classificação sempre vinha vazia) e
	// qwen2.5-coder:3b (sem esse problema de parsing, mas errou a classificação
	// multi-hop — é um modelo de código, não generalista). granite4:micro-h acertou
	// os dois casos de teste sem esses problemas.
	defaultRouterModel = "granite4:micro-h"
	graphQueryHops     = 2
	treeMaxDepth       = 3
)

// Config configura o Router e as quatro stores que ele orquestra. Diferente
// das fases anteriores, as quatro são obrigatórias neste MVP — não há
// configuração parcial (ex: só vector+graph sem tree/sql); use os pacotes
// individualmente se não precisar do roteamento completo.
type Config struct {
	Vector rag.Config
	Graph  graph.Config
	Tree   tree.Config
	SQL    sql.Config
	// RouterModel: modelo Ollama para classificação da pergunta. Vazio usa
	// defaultRouterModel — uma tarefa de classificação não precisa do mesmo
	// modelo usado na extração de entidades do Graph RAG.
	RouterModel string
}

// QueryResult carrega o resultado nativo da store escolhida — só os campos
// referentes à Strategy vêm preenchidos. O Router não sintetiza resposta
// final em texto; isso é responsabilidade de uma camada futura (Answer Reasoner).
type QueryResult struct {
	Strategy Strategy
	Vector   []rag.SearchResult
	Graph    []graph.Relation
	Tree     []tree.Node
	SQLRows  []map[string]any
	SQLQuery string // SQL gerado, preenchido só quando Strategy == StrategySQL
}

type classifyFunc func(ctx context.Context, prompt string) (string, error)

// Router é o ponto de entrada: uma pergunta, uma estratégia escolhida.
type Router struct {
	vector   *rag.UnifiedRAG
	graph    *graph.GraphStore
	tree     *tree.Tree
	sql      *sql.Store
	classify classifyFunc
}

// New cria um Router a partir de Config, iniciando as quatro stores.
func New(cfg Config) (*Router, error) {
	v, err := rag.New(cfg.Vector)
	if err != nil {
		return nil, fmt.Errorf("router: iniciar vector rag: %w", err)
	}
	g, err := graph.New(cfg.Graph)
	if err != nil {
		return nil, fmt.Errorf("router: iniciar graph rag: %w", err)
	}
	tr, err := tree.New(cfg.Tree)
	if err != nil {
		return nil, fmt.Errorf("router: iniciar tree rag: %w", err)
	}
	sq, err := sql.New(cfg.SQL)
	if err != nil {
		return nil, fmt.Errorf("router: iniciar sql: %w", err)
	}

	model := cfg.RouterModel
	if model == "" {
		model = defaultRouterModel
	}
	classify := func(ctx context.Context, prompt string) (string, error) {
		return ollama.Complete(ctx, "", model, prompt, false)
	}

	return newRouterWithClassifier(v, g, tr, sq, classify), nil
}

// newRouterWithClassifier permite injetar um classifyFunc fake em testes,
// sem depender de Ollama rodando.
func newRouterWithClassifier(v *rag.UnifiedRAG, g *graph.GraphStore, tr *tree.Tree, sq *sql.Store, classify classifyFunc) *Router {
	return &Router{vector: v, graph: g, tree: tr, sql: sq, classify: classify}
}

// AddDocuments indexa os documentos em vector e graph (fan-out) — o Router não
// decide antecipadamente qual store vai responder, então ambas precisam ter
// os dados disponíveis na hora da Query. tree e sql não usam esse formato de
// documento: ver AddMarkdownDocument e a introspecção de schema em pkg/sql.
func (r *Router) AddDocuments(ctx context.Context, docs []rag.Document) error {
	if err := r.vector.AddDocuments(ctx, docs); err != nil {
		return fmt.Errorf("router: vector AddDocuments: %w", err)
	}
	if err := r.graph.AddDocuments(ctx, docs); err != nil {
		return fmt.Errorf("router: graph AddDocuments: %w", err)
	}
	return nil
}

// AddMarkdownDocument indexa um documento markdown na store tree (Vectorless
// RAG) — formato diferente de AddDocuments porque tree precisa da estrutura
// hierárquica do documento inteiro, não de uma lista de trechos já cortados.
func (r *Router) AddMarkdownDocument(ctx context.Context, id, title, rawMarkdown string) error {
	return r.tree.AddDocument(ctx, id, title, rawMarkdown)
}

const classifyPrompt = `Classifique a pergunta abaixo em "vector", "graph" ou "both":
- "vector": busca por similaridade de conteúdo/texto, factual, um único documento resolve.
- "graph": pergunta sobre relações entre entidades, multi-hop (ex: "quem trabalha em X e onde X fica").
- "both": a pergunta precisa das duas coisas ao mesmo tempo (ex: "quem trabalha na empresa X e o que a empresa X faz?").
- "tree": pergunta sobre uma seção específica de um documento estruturado (relatório, manual, guia com capítulos/seções).
- "sql": pergunta sobre dados tabulares/numéricos (contagens, médias, "quantos", "qual o maior/menor").
Responda apenas com a palavra: vector, graph, both, tree ou sql.

Exemplos:
Pergunta: O que é fotossíntese?
Resposta: vector

Pergunta: Quem dirigiu o filme que a atriz Ana Souza estrelou, e em que cidade esse diretor nasceu?
Resposta: graph

Pergunta: Quem é o diretor do filme Nebulosa e o que é ficção científica?
Resposta: both

Pergunta: O que o manual diz sobre manutenção do motor?
Resposta: tree

Pergunta: Quantos funcionários ganham mais de 5000 por mês?
Resposta: sql

Pergunta: %s
Resposta:`

// Query classifica a pergunta e despacha para a(s) store(s) escolhida(s).
func (r *Router) Query(ctx context.Context, question string, topK int) (QueryResult, error) {
	switch r.classifyStrategy(ctx, question) {
	case StrategyGraph:
		results, err := r.graph.Query(ctx, question, graphQueryHops)
		if err != nil {
			return QueryResult{}, fmt.Errorf("router: graph query: %w", err)
		}
		return QueryResult{Strategy: StrategyGraph, Graph: results}, nil
	case StrategyBoth:
		vecResults, graphResults, err := r.queryBoth(ctx, question, topK)
		if err != nil {
			return QueryResult{}, err
		}
		return QueryResult{Strategy: StrategyBoth, Vector: vecResults, Graph: graphResults}, nil
	case StrategyTree:
		results, err := r.tree.Query(ctx, question, treeMaxDepth)
		if err != nil {
			return QueryResult{}, fmt.Errorf("router: tree query: %w", err)
		}
		return QueryResult{Strategy: StrategyTree, Tree: results}, nil
	case StrategySQL:
		rows, generatedSQL, err := r.sql.Query(ctx, question)
		if err != nil {
			return QueryResult{}, fmt.Errorf("router: sql query: %w", err)
		}
		return QueryResult{Strategy: StrategySQL, SQLRows: rows, SQLQuery: generatedSQL}, nil
	default:
		results, err := r.vector.Query(ctx, question, topK)
		if err != nil {
			return QueryResult{}, fmt.Errorf("router: vector query: %w", err)
		}
		return QueryResult{Strategy: StrategyVector, Vector: results}, nil
	}
}

// FusedResult carrega resultados de vector e graph juntos, sem tentar unificar
// num único score: não existe métrica de relevância comparável entre a
// similaridade de embedding do Vector RAG e uma aresta do Graph RAG — inventar
// uma seria falsa precisão. "Fusão" aqui é "roda os dois, devolve os dois", não
// "um ranking único misturado".
type FusedResult struct {
	Vector []rag.SearchResult
	Graph  []graph.Relation
}

// QueryFused roda vector e graph em paralelo, sem passar pela classificação, e
// devolve os dois resultados. Uso explícito — Query (dispatch único) continua o
// caminho padrão, mais barato.
func (r *Router) QueryFused(ctx context.Context, question string, topK int) (FusedResult, error) {
	vecResults, graphResults, err := r.queryBoth(ctx, question, topK)
	if err != nil {
		return FusedResult{}, err
	}
	return FusedResult{Vector: vecResults, Graph: graphResults}, nil
}

// queryBoth roda vector e graph em paralelo — usado tanto por QueryFused (sempre)
// quanto por Query quando a classificação vier "both".
func (r *Router) queryBoth(ctx context.Context, question string, topK int) ([]rag.SearchResult, []graph.Relation, error) {
	var (
		vecResults   []rag.SearchResult
		graphResults []graph.Relation
		vecErr       error
		graphErr     error
	)

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		vecResults, vecErr = r.vector.Query(ctx, question, topK)
	}()
	go func() {
		defer wg.Done()
		graphResults, graphErr = r.graph.Query(ctx, question, graphQueryHops)
	}()
	wg.Wait()

	if vecErr != nil {
		return nil, nil, fmt.Errorf("router: vector query: %w", vecErr)
	}
	if graphErr != nil {
		return nil, nil, fmt.Errorf("router: graph query: %w", graphErr)
	}

	return vecResults, graphResults, nil
}

// classifyStrategy nunca deixa uma classificação ruim ou indisponível travar a
// Query inteira — cai para StrategyVector (fallback seguro e mais barato; uma
// resposta ambígua/não-parseável não vira StrategyBoth automaticamente — "both"
// só acontece quando o LLM pede explicitamente).
func (r *Router) classifyStrategy(ctx context.Context, question string) Strategy {
	raw, err := r.classify(ctx, fmt.Sprintf(classifyPrompt, question))
	if err != nil {
		return StrategyVector
	}
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case string(StrategyGraph):
		return StrategyGraph
	case string(StrategyBoth):
		return StrategyBoth
	case string(StrategyTree):
		return StrategyTree
	case string(StrategySQL):
		return StrategySQL
	default:
		return StrategyVector
	}
}
