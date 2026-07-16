package sql

import (
	"context"
	"strings"
	"testing"
)

func newTestStore(t *testing.T, generate generateFunc) *Store {
	t.Helper()
	s, err := newStoreWithGenerator(":memory:", generate)
	if err != nil {
		t.Fatalf("newStoreWithGenerator: %v", err)
	}
	t.Cleanup(func() { s.db.Close() })

	if _, err := s.db.Exec(`CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)`); err != nil {
		t.Fatalf("criar tabela: %v", err)
	}
	if _, err := s.db.Exec(`INSERT INTO users (name, age) VALUES ('Ana', 30), ('Bruno', 25)`); err != nil {
		t.Fatalf("inserir dados: %v", err)
	}

	// Reintrospecta: a introspecção original (em newStoreWithGenerator) rodou
	// antes de a tabela existir.
	schema, err := introspectSchema(s.db)
	if err != nil {
		t.Fatalf("introspectSchema: %v", err)
	}
	s.schema = schema
	return s
}

func TestStoreIntrospectsSchema(t *testing.T) {
	s := newTestStore(t, nil)
	if s.schema == "" {
		t.Fatal("esperava schema não vazio após CREATE TABLE")
	}
	if !strings.Contains(s.schema, "users") || !strings.Contains(s.schema, "name") || !strings.Contains(s.schema, "age") {
		t.Errorf("schema não contém tabela/colunas esperadas: %q", s.schema)
	}
}

func TestStoreQueryExecutesGeneratedSQL(t *testing.T) {
	generate := func(_ context.Context, _ string) (string, error) {
		return "SELECT name, age FROM users WHERE age > 26", nil
	}
	s := newTestStore(t, generate)

	rows, sqlText, err := s.Query(context.Background(), "quem tem mais de 26 anos?")
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if sqlText != "SELECT name, age FROM users WHERE age > 26" {
		t.Errorf("sql inesperado: %q", sqlText)
	}
	if len(rows) != 1 || rows[0]["name"] != "Ana" {
		t.Errorf("resultado inesperado: %+v", rows)
	}
}

func TestStoreQueryRejectsUnsafeGeneratedSQL(t *testing.T) {
	generate := func(_ context.Context, _ string) (string, error) {
		return "DROP TABLE users", nil
	}
	s := newTestStore(t, generate)

	_, _, err := s.Query(context.Background(), "apague tudo")
	if err == nil {
		t.Fatal("esperava erro para SQL destrutivo gerado, obteve nil")
	}

	// confirma que a tabela realmente não foi apagada (defesa em profundidade
	// do teste: o erro sozinho não prova que a query nunca rodou).
	var count int
	if err := s.db.QueryRow("SELECT COUNT(*) FROM users").Scan(&count); err != nil {
		t.Fatalf("tabela users não deveria ter sido afetada: %v", err)
	}
	if count != 2 {
		t.Errorf("esperava 2 linhas intactas em users, obtido %d", count)
	}
}
