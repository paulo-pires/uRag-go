package persist

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	_ "github.com/jackc/pgx/v5/stdlib"
	_ "modernc.org/sqlite"
)

type DBConn interface {
	Exec(query string, args ...any) (sql.Result, error)
	Query(query string, args ...any) (*sql.Rows, error)
	QueryRow(query string, args ...any) *sql.Row
	ExecContext(ctx context.Context, query string, args ...any) (sql.Result, error)
	QueryContext(ctx context.Context, query string, args ...any) (*sql.Rows, error)
	QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row
	BeginTx(ctx context.Context, opts *sql.TxOptions) (TxConn, error)
}

type TxConn interface {
	Exec(query string, args ...any) (sql.Result, error)
	Query(query string, args ...any) (*sql.Rows, error)
	QueryRow(query string, args ...any) *sql.Row
	ExecContext(ctx context.Context, query string, args ...any) (sql.Result, error)
	QueryContext(ctx context.Context, query string, args ...any) (*sql.Rows, error)
	QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row
	Commit() error
	Rollback() error
}

type wrappedDB struct {
	db     *sql.DB
	dbType string
}

func (w *wrappedDB) query(q string) string {
	if w.dbType != "postgres" {
		return q
	}
	// Postgres OR REPLACE / IGNORE translations
	if strings.Contains(q, "INSERT OR IGNORE INTO graph_metadata") {
		q = strings.ReplaceAll(q, "INSERT OR IGNORE INTO graph_metadata", "INSERT INTO graph_metadata")
		q = q + " ON CONFLICT (key) DO NOTHING"
	}
	if strings.Contains(q, "INSERT OR REPLACE INTO graph_metadata") {
		q = strings.ReplaceAll(q, "INSERT OR REPLACE INTO graph_metadata", "INSERT INTO graph_metadata")
		q = q + " ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP"
	}

	var sb strings.Builder
	paramIndex := 1
	for {
		idx := strings.Index(q, "?")
		if idx == -1 {
			sb.WriteString(q)
			break
		}
		sb.WriteString(q[:idx])
		sb.WriteString("$")
		sb.WriteString(strconv.Itoa(paramIndex))
		paramIndex++
		q = q[idx+1:]
	}
	return sb.String()
}

func (w *wrappedDB) Exec(query string, args ...any) (sql.Result, error) {
	return w.db.Exec(w.query(query), args...)
}

func (w *wrappedDB) Query(query string, args ...any) (*sql.Rows, error) {
	return w.db.Query(w.query(query), args...)
}

func (w *wrappedDB) QueryRow(query string, args ...any) *sql.Row {
	return w.db.QueryRow(w.query(query), args...)
}

func (w *wrappedDB) ExecContext(ctx context.Context, query string, args ...any) (sql.Result, error) {
	return w.db.ExecContext(ctx, w.query(query), args...)
}

func (w *wrappedDB) QueryContext(ctx context.Context, query string, args ...any) (*sql.Rows, error) {
	return w.db.QueryContext(ctx, w.query(query), args...)
}

func (w *wrappedDB) QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row {
	return w.db.QueryRowContext(ctx, w.query(query), args...)
}

func (w *wrappedDB) BeginTx(ctx context.Context, opts *sql.TxOptions) (TxConn, error) {
	tx, err := w.db.BeginTx(ctx, opts)
	if err != nil {
		return nil, err
	}
	return &wrappedTx{tx: tx, dbType: w.dbType}, nil
}

type wrappedTx struct {
	tx     *sql.Tx
	dbType string
}

func (w *wrappedTx) query(q string) string {
	if w.dbType != "postgres" {
		return q
	}
	// Postgres OR REPLACE / IGNORE translations inside transaction
	if strings.Contains(q, "INSERT OR IGNORE INTO graph_metadata") {
		q = strings.ReplaceAll(q, "INSERT OR IGNORE INTO graph_metadata", "INSERT INTO graph_metadata")
		q = q + " ON CONFLICT (key) DO NOTHING"
	}
	if strings.Contains(q, "INSERT OR REPLACE INTO graph_metadata") {
		q = strings.ReplaceAll(q, "INSERT OR REPLACE INTO graph_metadata", "INSERT INTO graph_metadata")
		q = q + " ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP"
	}

	var sb strings.Builder
	paramIndex := 1
	for {
		idx := strings.Index(q, "?")
		if idx == -1 {
			sb.WriteString(q)
			break
		}
		sb.WriteString(q[:idx])
		sb.WriteString("$")
		sb.WriteString(strconv.Itoa(paramIndex))
		paramIndex++
		q = q[idx+1:]
	}
	return sb.String()
}

func (w *wrappedTx) Exec(query string, args ...any) (sql.Result, error) {
	return w.tx.Exec(w.query(query), args...)
}

func (w *wrappedTx) Query(query string, args ...any) (*sql.Rows, error) {
	return w.tx.Query(w.query(query), args...)
}

func (w *wrappedTx) QueryRow(query string, args ...any) *sql.Row {
	return w.tx.QueryRow(w.query(query), args...)
}

func (w *wrappedTx) ExecContext(ctx context.Context, query string, args ...any) (sql.Result, error) {
	return w.tx.ExecContext(ctx, w.query(query), args...)
}

func (w *wrappedTx) QueryContext(ctx context.Context, query string, args ...any) (*sql.Rows, error) {
	return w.tx.QueryContext(ctx, w.query(query), args...)
}

func (w *wrappedTx) QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row {
	return w.tx.QueryRowContext(ctx, w.query(query), args...)
}

func (w *wrappedTx) Commit() error {
	return w.tx.Commit()
}

func (w *wrappedTx) Rollback() error {
	return w.tx.Rollback()
}

// Store gerencia a persistência do grafo
type Store struct {
	db     DBConn
	schema *Schema
	cache  *GraphCache // Cache em memória para operações rápidas
}

// Config configuração da store
type Config struct {
	DBType    string        // Tipo do banco (sqlite ou postgres)
	DSN       string        // Data Source Name (ex: "file:graph.db" ou postgres connection string)
	CacheSize int           // Tamanho do cache em memória
	CacheTTL  time.Duration // TTL do cache
}

// NewStore cria uma nova store
func NewStore(cfg Config) (*Store, error) {
	if cfg.DBType == "" {
		cfg.DBType = "sqlite"
	}

	var db *sql.DB
	var err error
	if cfg.DBType == "postgres" {
		db, err = sql.Open("pgx", cfg.DSN)
		if err != nil {
			return nil, fmt.Errorf("abrir banco postgres: %w", err)
		}
		db.SetMaxOpenConns(50)
	} else {
		db, err = sql.Open("sqlite", cfg.DSN)
		if err != nil {
			return nil, fmt.Errorf("abrir banco sqlite: %w", err)
		}
		db.SetMaxOpenConns(1)
		db.SetMaxIdleConns(1)
	}

	// Testa conexão
	if err := db.Ping(); err != nil {
		db.Close()
		return nil, fmt.Errorf("ping banco: %w", err)
	}

	w := &wrappedDB{db: db, dbType: cfg.DBType}
	store := &Store{
		db:     w,
		schema: NewSchema(w, cfg.DBType),
		cache:  NewGraphCache(cfg.CacheSize, cfg.CacheTTL),
	}

	// Cria schema se não existir
	if err := store.schema.CreateTables(); err != nil {
		db.Close()
		return nil, fmt.Errorf("criar schema: %w", err)
	}

	if err := store.schema.Migrate(); err != nil {
		db.Close()
		return nil, fmt.Errorf("migrar schema: %w", err)
	}

	return store, nil
}

// Close fecha a conexão com o banco
func (s *Store) Close() error {
	if w, ok := s.db.(*wrappedDB); ok {
		return w.db.Close()
	}
	return nil
}


// AddEntity adiciona ou atualiza uma entidade
func (s *Store) AddEntity(ctx context.Context, entity *Entity) error {
	// Serializa properties para JSON
	propsJSON, err := json.Marshal(entity.Properties)
	if err != nil {
		return fmt.Errorf("serializar properties: %w", err)
	}

	query := `
        INSERT INTO entities (id, name, type, properties, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            type = excluded.type,
            properties = excluded.properties,
            updated_at = CURRENT_TIMESTAMP
    `

	_, err = s.db.ExecContext(ctx, query, entity.ID, entity.Name, entity.Type, string(propsJSON))
	if err != nil {
		return fmt.Errorf("inserir entidade: %w", err)
	}

	// Atualiza cache
	s.cache.SetEntity(entity.ID, entity)

	return nil
}

// GetEntity busca uma entidade por ID
func (s *Store) GetEntity(ctx context.Context, id string) (*Entity, error) {
	// Tenta cache primeiro
	if entity := s.cache.GetEntity(id); entity != nil {
		return entity, nil
	}

	query := `SELECT id, name, type, properties, created_at, updated_at FROM entities WHERE id = ?`
	row := s.db.QueryRowContext(ctx, query, id)

	var entityRow EntityRow
	err := row.Scan(&entityRow.ID, &entityRow.Name, &entityRow.Type,
		&entityRow.Properties, &entityRow.CreatedAt, &entityRow.UpdatedAt)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil // Entidade não encontrada
		}
		return nil, fmt.Errorf("buscar entidade: %w", err)
	}

	entity, err := entityRow.ToEntity()
	if err != nil {
		return nil, err
	}

	// Atualiza cache
	s.cache.SetEntity(id, entity)

	return entity, nil
}

// AddRelation adiciona uma relação
func (s *Store) AddRelation(ctx context.Context, rel *Relation) error {
	// Verifica se entidades existem
	if _, err := s.GetEntity(ctx, rel.SourceID); err != nil {
		return fmt.Errorf("entidade source não existe: %w", err)
	}
	if _, err := s.GetEntity(ctx, rel.TargetID); err != nil {
		return fmt.Errorf("entidade target não existe: %w", err)
	}

	// Serializa properties
	propsJSON, err := json.Marshal(rel.Properties)
	if err != nil {
		return fmt.Errorf("serializar properties: %w", err)
	}

	query := `
        INSERT INTO relations (id, source_id, target_id, type, properties, created_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(id) DO UPDATE SET
            source_id = excluded.source_id,
            target_id = excluded.target_id,
            type = excluded.type,
            properties = excluded.properties
    `

	_, err = s.db.ExecContext(ctx, query, rel.ID, rel.SourceID, rel.TargetID, rel.Type, string(propsJSON))
	if err != nil {
		return fmt.Errorf("inserir relação: %w", err)
	}

	// Invalida cache após modificações
	s.cache.Clear()

	return nil
}

// GetRelations busca relações de uma entidade
func (s *Store) GetRelations(ctx context.Context, entityID string, relationType string) ([]*Relation, error) {
	cacheKey := fmt.Sprintf("%s:%s", entityID, relationType)
	if cached := s.cache.GetRelations(cacheKey); cached != nil {
		return cached, nil
	}

	query := `
        SELECT id, source_id, target_id, type, properties, created_at
        FROM relations
        WHERE source_id = ? AND type = ?
    `

	rows, err := s.db.QueryContext(ctx, query, entityID, relationType)
	if err != nil {
		return nil, fmt.Errorf("buscar relações: %w", err)
	}
	defer rows.Close()

	var relations []*Relation
	for rows.Next() {
		var row RelationRow
		err := rows.Scan(&row.ID, &row.SourceID, &row.TargetID, &row.Type,
			&row.Properties, &row.CreatedAt)
		if err != nil {
			return nil, fmt.Errorf("scan relação: %w", err)
		}

		rel, err := row.ToRelation()
		if err != nil {
			return nil, err
		}
		relations = append(relations, rel)
	}

	// Atualiza cache
	s.cache.SetRelations(cacheKey, relations)

	return relations, nil
}

// LoadFullGraph carrega o grafo completo
func (s *Store) LoadFullGraph(ctx context.Context) (*GraphSnapshot, error) {
	// Carrega entidades
	entityRows, err := s.db.QueryContext(ctx, `SELECT id, name, type, properties, created_at, updated_at FROM entities`)
	if err != nil {
		return nil, fmt.Errorf("carregar entidades: %w", err)
	}
	defer entityRows.Close()

	var entities []Entity
	for entityRows.Next() {
		var row EntityRow
		err := entityRows.Scan(&row.ID, &row.Name, &row.Type, &row.Properties, &row.CreatedAt, &row.UpdatedAt)
		if err != nil {
			return nil, fmt.Errorf("scan entidade: %w", err)
		}

		entity, err := row.ToEntity()
		if err != nil {
			return nil, err
		}
		entities = append(entities, *entity)
	}

	// Carrega relações
	relationRows, err := s.db.QueryContext(ctx, `SELECT id, source_id, target_id, type, properties, created_at FROM relations`)
	if err != nil {
		return nil, fmt.Errorf("carregar relações: %w", err)
	}
	defer relationRows.Close()

	var relations []Relation
	for relationRows.Next() {
		var row RelationRow
		err := relationRows.Scan(&row.ID, &row.SourceID, &row.TargetID, &row.Type, &row.Properties, &row.CreatedAt)
		if err != nil {
			return nil, fmt.Errorf("scan relação: %w", err)
		}

		rel, err := row.ToRelation()
		if err != nil {
			return nil, err
		}
		relations = append(relations, *rel)
	}

	// Carrega versão
	var version string
	err = s.db.QueryRowContext(ctx, `SELECT value FROM graph_metadata WHERE key = 'schema_version'`).Scan(&version)
	if err != nil && err != sql.ErrNoRows {
		return nil, fmt.Errorf("carregar versão: %w", err)
	}

	return &GraphSnapshot{
		Entities:  entities,
		Relations: relations,
		Version:   version,
		Timestamp: time.Now(),
	}, nil
}

// SaveFullGraph salva o grafo completo (útil para backup/restore)
func (s *Store) SaveFullGraph(ctx context.Context, snapshot *GraphSnapshot) error {
	// Inicia transação para consistência
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("iniciar transação: %w", err)
	}
	defer tx.Rollback()

	// Limpa dados existentes
	if _, err := tx.ExecContext(ctx, `DELETE FROM relations`); err != nil {
		return fmt.Errorf("limpar relações: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM entities`); err != nil {
		return fmt.Errorf("limpar entidades: %w", err)
	}

	// Insere entidades
	for _, entity := range snapshot.Entities {
		propsJSON, _ := json.Marshal(entity.Properties)
		_, err := tx.ExecContext(ctx, `
            INSERT INTO entities (id, name, type, properties, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        `, entity.ID, entity.Name, entity.Type, string(propsJSON), entity.CreatedAt, entity.UpdatedAt)
		if err != nil {
			return fmt.Errorf("inserir entidade %s: %w", entity.ID, err)
		}
	}

	// Insere relações
	for _, rel := range snapshot.Relations {
		propsJSON, _ := json.Marshal(rel.Properties)
		_, err := tx.ExecContext(ctx, `
            INSERT INTO relations (id, source_id, target_id, type, properties, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        `, rel.ID, rel.SourceID, rel.TargetID, rel.Type, string(propsJSON), rel.CreatedAt)
		if err != nil {
			return fmt.Errorf("inserir relação %s: %w", rel.ID, err)
		}
	}

	// Atualiza versão
	_, err = tx.ExecContext(ctx, `
        INSERT OR REPLACE INTO graph_metadata (key, value, updated_at)
        VALUES ('schema_version', ?, CURRENT_TIMESTAMP)
    `, snapshot.Version)
	if err != nil {
		return fmt.Errorf("atualizar versão: %w", err)
	}

	// Commit
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit transação: %w", err)
	}

	// Limpa cache
	s.cache.Clear()

	return nil
}
