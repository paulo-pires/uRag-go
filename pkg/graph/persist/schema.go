package persist

import (
	"database/sql"

	_ "modernc.org/sqlite"
)

// Schema representa o schema do banco de dados
type Schema struct {
	db *sql.DB
}

// NewSchema cria um novo schema
func NewSchema(db *sql.DB) *Schema {
	return &Schema{db: db}
}

// CreateTables cria as tabelas necessárias
func (s *Schema) CreateTables() error {
	queries := []string{
		// Tabela de entidades
		`CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT,
            properties JSON,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )`,

		// Índice para busca por nome
		`CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)`,

		// Índice para busca por tipo
		`CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)`,

		// Tabela de relações
		`CREATE TABLE IF NOT EXISTS relations (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            type TEXT NOT NULL,
            properties JSON,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_id) REFERENCES entities(id) ON DELETE CASCADE,
            FOREIGN KEY (target_id) REFERENCES entities(id) ON DELETE CASCADE
        )`,

		// Índices para relações
		`CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id)`,
		`CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id)`,
		`CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(type)`,

		// Índice composto para consultas rápidas
		`CREATE INDEX IF NOT EXISTS idx_relations_source_type ON relations(source_id, type)`,

		// Tabela de metadados do grafo
		`CREATE TABLE IF NOT EXISTS graph_metadata (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )`,
	}

	for _, query := range queries {
		if _, err := s.db.Exec(query); err != nil {
			return err
		}
	}

	return nil
}

// Migrate realiza migrações futuras se necessário
func (s *Schema) Migrate() error {
	// Versão atual do schema
	_, err := s.db.Exec(`
        INSERT OR IGNORE INTO graph_metadata (key, value) 
        VALUES ('schema_version', '1.0.0')
    `)
	return err
}
