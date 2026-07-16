// Package sql implementa o Text-to-SQL do uRag-go: converte uma pergunta em
// linguagem natural numa query SQL SELECT, executa contra SQLite e devolve o
// resultado. Só SQLite no MVP (via modernc.org/sqlite, pure Go, sem CGO).
package sql

import (
	"context"
	"database/sql"
	"fmt"
	"regexp"
	"strings"

	_ "modernc.org/sqlite"

	"urag-go/internal/ollama"
	"urag-go/internal/openai"
)

// Config configura o Store.
type Config struct {
	// DSN: caminho do arquivo SQLite, ou ":memory:" para in-memory.
	DSN string
	// LLMProvider: "ollama" ou "openai" (compatível: OpenAI e providers que implementem o mesmo formato).
	LLMProvider string
	LLMModel    string
	// LLMBaseURL: override do endpoint. "" = default do provider (Ollama local,
	// ou api.openai.com para "openai") — obrigatório apontar pra providers
	// OpenAI-compatíveis que não são a OpenAI oficial (vLLM, LM Studio, etc).
	LLMBaseURL string
	// LLMAPIKey: usado só quando LLMProvider="openai".
	LLMAPIKey string
}

type generateFunc func(ctx context.Context, prompt string) (string, error)

// Store é o ponto de entrada do Text-to-SQL.
type Store struct {
	db       *sql.DB
	schema   string
	generate generateFunc
}

// New cria um Store: conecta ao banco, faz introspecção do schema (tabelas +
// colunas) e configura a geração de SQL via LLM.
func New(cfg Config) (*Store, error) {
	var generate generateFunc
	switch cfg.LLMProvider {
	case "ollama":
		generate = func(ctx context.Context, prompt string) (string, error) {
			return ollama.Complete(ctx, cfg.LLMBaseURL, cfg.LLMModel, prompt, false)
		}
	case "openai":
		generate = func(ctx context.Context, prompt string) (string, error) {
			return openai.Complete(ctx, cfg.LLMBaseURL, cfg.LLMAPIKey, cfg.LLMModel, prompt, false)
		}
	default:
		return nil, fmt.Errorf("sql: llm provider desconhecido: %q", cfg.LLMProvider)
	}
	return newStoreWithGenerator(cfg.DSN, generate)
}

// newStoreWithGenerator permite injetar um generateFunc fake em testes, sem
// depender de Ollama rodando — mas usa um banco SQLite real (in-memory via
// ":memory:"), já que introspecção de schema e execução são o cerne do pacote.
// NewWithGenerator cria um Store com uma função de geração de SQL já pronta,
// sem resolver via Config.LLMProvider/LLMModel — útil para geração customizada
// (provider fora de "ollama") ou para injetar um fake em testes.
func NewWithGenerator(dsn string, generate func(ctx context.Context, prompt string) (string, error)) (*Store, error) {
	return newStoreWithGenerator(dsn, generate)
}

func newStoreWithGenerator(dsn string, generate generateFunc) (*Store, error) {
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("sql: abrir banco: %w", err)
	}

	schema, err := introspectSchema(db)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("sql: introspecção de schema: %w", err)
	}

	return &Store{db: db, schema: schema, generate: generate}, nil
}

func introspectSchema(db *sql.DB) (string, error) {
	rows, err := db.Query(`SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'`)
	if err != nil {
		return "", err
	}
	defer rows.Close()

	var tables []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			return "", err
		}
		tables = append(tables, name)
	}
	if err := rows.Err(); err != nil {
		return "", err
	}

	var schema strings.Builder
	for _, table := range tables {
		cols, err := tableColumns(db, table)
		if err != nil {
			return "", err
		}
		fmt.Fprintf(&schema, "Tabela: %s (%s)\n", table, strings.Join(cols, ", "))
	}
	return schema.String(), nil
}

func tableColumns(db *sql.DB, table string) ([]string, error) {
	rows, err := db.Query(fmt.Sprintf("PRAGMA table_info(%s)", quoteIdent(table)))
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var cols []string
	for rows.Next() {
		var cid, notnull, pk int
		var name, ctype string
		var dflt any
		if err := rows.Scan(&cid, &name, &ctype, &notnull, &dflt, &pk); err != nil {
			return nil, err
		}
		cols = append(cols, fmt.Sprintf("%s %s", name, ctype))
	}
	return cols, rows.Err()
}

func quoteIdent(s string) string {
	return `"` + strings.ReplaceAll(s, `"`, `""`) + `"`
}

const generatePrompt = `Você tem acesso a um banco SQLite com o schema abaixo. Gere uma única query SQL
SELECT que responda à pergunta. Responda apenas com o SQL puro, sem explicação, sem markdown.

Schema:
%s

Pergunta: %s

SQL:`

// Close fecha a conexão com o banco.
func (s *Store) Close() error {
	return s.db.Close()
}

// Query gera SQL a partir da pergunta, valida (só leitura, 1 statement),
// executa e devolve as linhas junto com o SQL gerado (transparência: você vê
// o que rodou). A validação roda antes de qualquer execução, sem exceção.
func (s *Store) Query(ctx context.Context, question string) ([]map[string]any, string, error) {
	raw, err := s.generate(ctx, fmt.Sprintf(generatePrompt, s.schema, question))
	if err != nil {
		return nil, "", fmt.Errorf("sql: gerar sql: %w", err)
	}
	query := extractSQL(raw)

	if err := validateReadOnlySelect(query); err != nil {
		return nil, query, fmt.Errorf("sql: query gerada rejeitada: %w", err)
	}

	rows, err := s.db.QueryContext(ctx, query)
	if err != nil {
		return nil, query, fmt.Errorf("sql: executar query: %w", err)
	}
	defer rows.Close()

	results, err := scanRows(rows)
	if err != nil {
		return nil, query, fmt.Errorf("sql: ler resultado: %w", err)
	}
	return results, query, nil
}

// extractSQL remove cercas de código markdown (```sql ... ```) caso o LLM
// ignore a instrução de responder só com SQL puro.
func extractSQL(raw string) string {
	s := strings.TrimSpace(raw)
	if strings.HasPrefix(s, "```") {
		s = strings.TrimPrefix(s, "```sql")
		s = strings.TrimPrefix(s, "```")
		s = strings.TrimSuffix(s, "```")
		s = strings.TrimSpace(s)
	}
	return s
}

var forbiddenKeywords = regexp.MustCompile(`(?i)\b(insert|update|delete|drop|alter|attach|detach|pragma|create|replace|truncate|vacuum)\b`)

// validateReadOnlySelect é a barreira de segurança obrigatória entre o SQL
// gerado por LLM (entrada não confiável) e a execução real contra o banco:
// só SELECT/WITH, um único statement, sem palavras-chave de escrita/DDL.
func validateReadOnlySelect(query string) error {
	trimmed := strings.TrimSpace(query)
	if trimmed == "" {
		return fmt.Errorf("sql vazio")
	}

	nonEmptyStatements := 0
	for _, part := range strings.Split(trimmed, ";") {
		if strings.TrimSpace(part) != "" {
			nonEmptyStatements++
		}
	}
	if nonEmptyStatements > 1 {
		return fmt.Errorf("múltiplos statements não são permitidos")
	}

	upper := strings.ToUpper(trimmed)
	if !strings.HasPrefix(upper, "SELECT") && !strings.HasPrefix(upper, "WITH") {
		return fmt.Errorf("só SELECT (ou WITH ... SELECT) é permitido")
	}

	if forbiddenKeywords.MatchString(trimmed) {
		return fmt.Errorf("statement contém palavra-chave não permitida (só leitura)")
	}

	return nil
}

func scanRows(rows *sql.Rows) ([]map[string]any, error) {
	cols, err := rows.Columns()
	if err != nil {
		return nil, err
	}

	var results []map[string]any
	for rows.Next() {
		values := make([]any, len(cols))
		ptrs := make([]any, len(cols))
		for i := range values {
			ptrs[i] = &values[i]
		}
		if err := rows.Scan(ptrs...); err != nil {
			return nil, err
		}
		row := make(map[string]any, len(cols))
		for i, col := range cols {
			row[col] = values[i]
		}
		results = append(results, row)
	}
	return results, rows.Err()
}
