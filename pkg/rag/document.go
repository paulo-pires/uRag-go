package rag

// Document é a unidade de conteúdo indexada e retornada pelo UnifiedRAG.
type Document struct {
	ID      string
	Content string
	Source  string
	Meta    map[string]string
}

// SearchResult é um Document retornado por Query, com o score de similaridade.
type SearchResult struct {
	Document Document
	Score    float32
}
