package sql

import "testing"

func TestValidateReadOnlySelectAcceptsSelect(t *testing.T) {
	cases := []string{
		"SELECT * FROM users",
		"select id, name from users where id = 1",
		"WITH t AS (SELECT * FROM users) SELECT * FROM t",
		"SELECT * FROM users;",
	}
	for _, c := range cases {
		if err := validateReadOnlySelect(c); err != nil {
			t.Errorf("validateReadOnlySelect(%q) deveria aceitar, rejeitou: %v", c, err)
		}
	}
}

func TestValidateReadOnlySelectRejectsWriteAndDDL(t *testing.T) {
	cases := []string{
		"DROP TABLE users",
		"DELETE FROM users",
		"UPDATE users SET name = 'x'",
		"INSERT INTO users (name) VALUES ('x')",
		"ALTER TABLE users ADD COLUMN x TEXT",
		"CREATE TABLE evil (id INTEGER)",
		"PRAGMA writable_schema = 1",
		"ATTACH DATABASE 'x.db' AS x",
	}
	for _, c := range cases {
		if err := validateReadOnlySelect(c); err == nil {
			t.Errorf("validateReadOnlySelect(%q) deveria rejeitar, aceitou", c)
		}
	}
}

func TestValidateReadOnlySelectRejectsStackedStatements(t *testing.T) {
	cases := []string{
		"SELECT * FROM users; DROP TABLE users",
		"SELECT * FROM users; DELETE FROM users;",
	}
	for _, c := range cases {
		if err := validateReadOnlySelect(c); err == nil {
			t.Errorf("validateReadOnlySelect(%q) deveria rejeitar (statements empilhados), aceitou", c)
		}
	}
}

func TestValidateReadOnlySelectRejectsEmpty(t *testing.T) {
	if err := validateReadOnlySelect(""); err == nil {
		t.Error("validateReadOnlySelect(\"\") deveria rejeitar, aceitou")
	}
	if err := validateReadOnlySelect("   "); err == nil {
		t.Error("validateReadOnlySelect(espaços) deveria rejeitar, aceitou")
	}
}

func TestExtractSQLStripsMarkdownFences(t *testing.T) {
	cases := map[string]string{
		"```sql\nSELECT * FROM users\n```": "SELECT * FROM users",
		"```\nSELECT 1\n```":               "SELECT 1",
		"SELECT 1":                         "SELECT 1",
	}
	for input, want := range cases {
		got := extractSQL(input)
		if got != want {
			t.Errorf("extractSQL(%q) = %q, esperado %q", input, got, want)
		}
	}
}
