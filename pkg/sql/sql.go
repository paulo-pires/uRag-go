// Package sql implementa o Text-to-SQL do uRag-go: converte uma pergunta em
// linguagem natural numa query SQL SELECT, executa contra SQLite e devolve o
// resultado. Só SQLite no MVP (via modernc.org/sqlite, pure Go, sem CGO).
package sql

import (
	"context"
	"database/sql"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"sync"

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
	mu       sync.RWMutex
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
	s.mu.RLock()
	schema := s.schema
	s.mu.RUnlock()

	raw, err := s.generate(ctx, fmt.Sprintf(generatePrompt, schema, question))
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

// LoadData imports tables from CSV or JSON format.
func (s *Store) LoadData(ctx context.Context, tableName string, format string, rawData string) error {
	// Sanitize tableName: only alphanumeric and underscores
	re := regexp.MustCompile(`^[a-zA-Z0-9_]+$`)
	if !re.MatchString(tableName) {
		return fmt.Errorf("nome de tabela inválido: deve conter apenas letras, números e underscores")
	}

	var headers []string
	var rows [][]any

	switch strings.ToLower(format) {
	case "csv":
		reader := csv.NewReader(strings.NewReader(rawData))
		records, err := reader.ReadAll()
		if err != nil {
			return fmt.Errorf("erro ao ler CSV: %w", err)
		}
		if len(records) == 0 {
			return fmt.Errorf("CSV vazio")
		}
		// Headers
		for _, h := range records[0] {
			cleanH := strings.TrimSpace(h)
			if !re.MatchString(cleanH) {
				cleanH = regexp.MustCompile(`[^a-zA-Z0-9_]`).ReplaceAllString(cleanH, "_")
				if cleanH == "" || !re.MatchString(cleanH) {
					cleanH = "col_" + cleanH
					cleanH = regexp.MustCompile(`[^a-zA-Z0-9_]`).ReplaceAllString(cleanH, "_")
				}
			}
			headers = append(headers, cleanH)
		}
		// Data rows
		for _, r := range records[1:] {
			row := make([]any, len(headers))
			for i, val := range r {
				if i < len(headers) {
					row[i] = val
				}
			}
			rows = append(rows, row)
		}

	case "json":
		var records []map[string]any
		dec := json.NewDecoder(strings.NewReader(rawData))
		if err := dec.Decode(&records); err != nil {
			// Try decoding as single object
			var single map[string]any
			if err2 := json.Unmarshal([]byte(rawData), &single); err2 == nil {
				records = []map[string]any{single}
			} else {
				return fmt.Errorf("erro ao ler JSON: %w", err)
			}
		}
		if len(records) == 0 {
			return fmt.Errorf("JSON vazio")
		}
		// Collect all unique keys across records to form stable column headers
		headerMap := make(map[string]bool)
		for _, rec := range records {
			for k := range rec {
				headerMap[k] = true
			}
		}
		for h := range headerMap {
			cleanH := strings.TrimSpace(h)
			if !re.MatchString(cleanH) {
				cleanH = regexp.MustCompile(`[^a-zA-Z0-9_]`).ReplaceAllString(cleanH, "_")
				if cleanH == "" || !re.MatchString(cleanH) {
					cleanH = "col_" + cleanH
					cleanH = regexp.MustCompile(`[^a-zA-Z0-9_]`).ReplaceAllString(cleanH, "_")
				}
			}
			headers = append(headers, cleanH)
		}

		// Map objects to rows
		for _, rec := range records {
			row := make([]any, len(headers))
			for i, h := range headers {
				val, exists := rec[h]
				if !exists {
					// try with clean replacement check or check original key
					for origK := range rec {
						cleanOrig := regexp.MustCompile(`[^a-zA-Z0-9_]`).ReplaceAllString(strings.TrimSpace(origK), "_")
						if cleanOrig == h {
							val = rec[origK]
							exists = true
							break
						}
					}
				}
				if exists {
					if val == nil {
						row[i] = nil
					} else {
						switch v := val.(type) {
						case string:
							row[i] = v
						default:
							bytes, _ := json.Marshal(v)
							row[i] = string(bytes)
						}
					}
				} else {
					row[i] = nil
				}
			}
			rows = append(rows, row)
		}

	default:
		return fmt.Errorf("formato desconhecido: %q. Use 'csv' ou 'json'", format)
	}

	if len(headers) == 0 {
		return fmt.Errorf("nenhuma coluna identificada para criação da tabela")
	}

	// Create table
	var colDecls []string
	for _, h := range headers {
		colDecls = append(colDecls, fmt.Sprintf("%s TEXT", quoteIdent(h)))
	}
	createSQL := fmt.Sprintf("CREATE TABLE IF NOT EXISTS %s (%s)", quoteIdent(tableName), strings.Join(colDecls, ", "))

	s.mu.Lock()
	defer s.mu.Unlock()

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("erro ao iniciar transação: %w", err)
	}
	defer tx.Rollback()

	if _, err := tx.ExecContext(ctx, createSQL); err != nil {
		return fmt.Errorf("erro ao criar tabela: %w", err)
	}

	// Insert rows
	var qMarks []string
	var colNames []string
	for _, h := range headers {
		colNames = append(colNames, quoteIdent(h))
		qMarks = append(qMarks, "?")
	}
	insertSQL := fmt.Sprintf("INSERT INTO %s (%s) VALUES (%s)", quoteIdent(tableName), strings.Join(colNames, ", "), strings.Join(qMarks, ", "))

	stmt, err := tx.PrepareContext(ctx, insertSQL)
	if err != nil {
		return fmt.Errorf("erro ao preparar insert: %w", err)
	}
	defer stmt.Close()

	for _, row := range rows {
		if _, err := stmt.ExecContext(ctx, row...); err != nil {
			return fmt.Errorf("erro ao inserir linha: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("erro ao commitar transação: %w", err)
	}

	// Recalculate schema
	schema, err := introspectSchema(s.db)
	if err != nil {
		return fmt.Errorf("erro ao atualizar schema: %w", err)
	}
	s.schema = schema

	return nil
}

