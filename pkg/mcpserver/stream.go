package mcpserver

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"urag-go/internal/ollama"
	"urag-go/internal/openai"
	"urag-go/pkg/router"
)

type StreamResponse struct {
	Type    string `json:"type"` // "thinking", "chunk", "source", "done"
	Content string `json:"content,omitempty"`
	Source  string `json:"source,omitempty"`
	Error   string `json:"error,omitempty"`
}

func (s *Server) handleStream(w http.ResponseWriter, r *http.Request) {
	// 1. Configura Headers SSE
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	writeEvent := func(evt StreamResponse) {
		data, _ := json.Marshal(evt)
		fmt.Fprintf(w, "data: %s\n\n", data)
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
	}

	q := r.URL.Query().Get("q")
	if q == "" {
		writeEvent(StreamResponse{Type: "done", Error: "parâmetro 'q' (pergunta) é obrigatório"})
		return
	}

	topK := 5
	if kStr := r.URL.Query().Get("k"); kStr != "" {
		if k, err := strconv.Atoi(kStr); err == nil && k > 0 {
			topK = k
		}
	}

	ctx := r.Context()

	// 2. Cria Router local para classificar e buscar
	// O classificador do router usa Ollama local (mesma decisão registrada)
	classify := func(c context.Context, prompt string) (string, error) {
		return ollama.Complete(c, s.config.LLMBaseURL, s.config.LLMModel, prompt, false)
	}

	rt := router.NewRouterWithClassifier(s.vector, s.graph, s.tree, s.sql, classify)

	writeEvent(StreamResponse{Type: "thinking", Content: "Classificando intenção da pergunta..."})

	// 3. Executa a query do Router para obter o contexto
	queryRes, err := rt.Query(ctx, q, topK)
	if err != nil {
		writeEvent(StreamResponse{Type: "done", Error: fmt.Sprintf("erro no roteamento: %v", err)})
		return
	}

	writeEvent(StreamResponse{Type: "thinking", Content: fmt.Sprintf("Estratégia escolhida: %s. Buscando contexto...", queryRes.Strategy)})

	// 4. Formata o contexto dependendo da estratégia escolhida
	var contextBuilder strings.Builder
	var sources []string

	switch queryRes.Strategy {
	case router.StrategyVector:
		for _, v := range queryRes.Vector {
			contextBuilder.WriteString(v.Document.Content)
			contextBuilder.WriteString("\n---\n")
			sources = append(sources, v.Document.ID)
		}
	case router.StrategyGraph:
		for _, rel := range queryRes.Graph {
			fmt.Fprintf(&contextBuilder, "%s --[%s]--> %s\n", rel.Source, rel.Relation, rel.Target)
			sources = append(sources, rel.DocID)
		}
	case router.StrategyBoth:
		contextBuilder.WriteString("Relações:\n")
		for _, rel := range queryRes.Graph {
			fmt.Fprintf(&contextBuilder, "%s --[%s]--> %s\n", rel.Source, rel.Relation, rel.Target)
			sources = append(sources, rel.DocID)
		}
		contextBuilder.WriteString("\nDocumentos:\n")
		for _, v := range queryRes.Vector {
			contextBuilder.WriteString(v.Document.Content)
			contextBuilder.WriteString("\n---\n")
			sources = append(sources, v.Document.ID)
		}
	case router.StrategyTree:
		for _, n := range queryRes.Tree {
			fmt.Fprintf(&contextBuilder, "%s: %s\n---\n", n.Title, n.Content)
			sources = append(sources, n.Title)
		}
	case router.StrategySQL:
		fmt.Fprintf(&contextBuilder, "Query SQL executada:\n%s\n\nResultados obtidos:\n", queryRes.SQLQuery)
		for _, row := range queryRes.SQLRows {
			rowJSON, _ := json.Marshal(row)
			contextBuilder.Write(rowJSON)
			contextBuilder.WriteString("\n")
		}
		sqlDSN := s.config.SQLDSN
		if sqlDSN == "" {
			sqlDSN = "urag_sql.db"
		}
		sources = append(sources, "Banco SQLite: "+sqlDSN)
	}

	contextStr := contextBuilder.String()
	if contextStr == "" {
		contextStr = "Nenhum contexto encontrado."
	}

	writeEvent(StreamResponse{Type: "thinking", Content: "Contexto recuperado. Sintetizando resposta..."})

	// 5. Formula prompt final para responder com base no contexto
	prompt := fmt.Sprintf(`Responda à pergunta do usuário baseando-se estritamente no contexto fornecido abaixo.
Se o contexto não for suficiente para responder, diga que não sabe. Não tente inventar informações.

Contexto:
%s

Pergunta:
%s

Resposta:`, contextStr, q)

	// 6. Executa a chamada do LLM em streaming
	onChunk := func(chunk string) {
		writeEvent(StreamResponse{Type: "chunk", Content: chunk})
	}

	if s.config.LLMProvider == "openai" {
		err = openai.StreamComplete(ctx, s.config.LLMBaseURL, s.config.LLMAPIKey, s.config.LLMModel, prompt, onChunk)
	} else {
		err = ollama.StreamComplete(ctx, s.config.LLMBaseURL, s.config.LLMModel, prompt, onChunk)
	}

	if err != nil {
		writeEvent(StreamResponse{Type: "done", Error: fmt.Sprintf("erro na síntese de resposta: %v", err)})
		return
	}

	// 7. Envia as fontes e finaliza
	for _, src := range sources {
		writeEvent(StreamResponse{Type: "source", Source: src})
	}

	writeEvent(StreamResponse{Type: "done"})
}
