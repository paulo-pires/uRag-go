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

func TestLoadData(t *testing.T) {
	s, err := newStoreWithGenerator(":memory:", nil)
	if err != nil {
		t.Fatalf("newStoreWithGenerator: %v", err)
	}
	t.Cleanup(func() { s.db.Close() })

	ctx := context.Background()

	// Test CSV load
	csvData := `name,city,age
Alice,New York,28
Bob,Paris,34`
	if err := s.LoadData(ctx, "people_csv", "csv", csvData); err != nil {
		t.Fatalf("LoadData CSV: %v", err)
	}

	// Verify people_csv table exists and has data
	var count int
	if err := s.db.QueryRow("SELECT COUNT(*) FROM people_csv").Scan(&count); err != nil {
		t.Fatalf("QueryRow CSV table: %v", err)
	}
	if count != 2 {
		t.Errorf("expected 2 rows in CSV table, got %d", count)
	}

	// Test JSON load
	jsonData := `[
		{"name": "Charlie", "city": "London", "age": 42},
		{"name": "Diana", "city": "Tokyo", "age": 19}
	]`
	if err := s.LoadData(ctx, "people_json", "json", jsonData); err != nil {
		t.Fatalf("LoadData JSON: %v", err)
	}

	// Verify people_json table exists and has data
	if err := s.db.QueryRow("SELECT COUNT(*) FROM people_json").Scan(&count); err != nil {
		t.Fatalf("QueryRow JSON table: %v", err)
	}
	if count != 2 {
		t.Errorf("expected 2 rows in JSON table, got %d", count)
	}

	// Verify schema is updated
	if !strings.Contains(s.schema, "people_csv") || !strings.Contains(s.schema, "people_json") {
		t.Errorf("schema did not update properly: %q", s.schema)
	}
}

