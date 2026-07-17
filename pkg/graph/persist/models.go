package persist

import (
	"database/sql"
	"encoding/json"
	"time"
)

// Entity representa uma entidade no grafo
type Entity struct {
	ID         string                 `json:"id"`
	Name       string                 `json:"name"`
	Type       string                 `json:"type"`
	Properties map[string]interface{} `json:"properties"`
	CreatedAt  time.Time              `json:"created_at"`
	UpdatedAt  time.Time              `json:"updated_at"`
}

// Relation representa uma relação entre entidades
type Relation struct {
	ID         string                 `json:"id"`
	SourceID   string                 `json:"source_id"`
	TargetID   string                 `json:"target_id"`
	Type       string                 `json:"type"`
	Properties map[string]interface{} `json:"properties"`
	CreatedAt  time.Time              `json:"created_at"`
}

// GraphSnapshot representa um snapshot completo do grafo
type GraphSnapshot struct {
	Entities  []Entity   `json:"entities"`
	Relations []Relation `json:"relations"`
	Version   string     `json:"version"`
	Timestamp time.Time  `json:"timestamp"`
}

// EntityRow para scan do SQL
type EntityRow struct {
	ID         string
	Name       string
	Type       sql.NullString
	Properties sql.NullString
	CreatedAt  time.Time
	UpdatedAt  time.Time
}

// RelationRow para scan do SQL
type RelationRow struct {
	ID         string
	SourceID   string
	TargetID   string
	Type       string
	Properties sql.NullString
	CreatedAt  time.Time
}

// ToEntity converte EntityRow para Entity
func (r *EntityRow) ToEntity() (*Entity, error) {
	entity := &Entity{
		ID:        r.ID,
		Name:      r.Name,
		CreatedAt: r.CreatedAt,
		UpdatedAt: r.UpdatedAt,
	}

	if r.Type.Valid {
		entity.Type = r.Type.String
	}

	if r.Properties.Valid && r.Properties.String != "" {
		var props map[string]interface{}
		if err := json.Unmarshal([]byte(r.Properties.String), &props); err != nil {
			return nil, err
		}
		entity.Properties = props
	}

	return entity, nil
}

// ToRelation converte RelationRow para Relation
func (r *RelationRow) ToRelation() (*Relation, error) {
	rel := &Relation{
		ID:        r.ID,
		SourceID:  r.SourceID,
		TargetID:  r.TargetID,
		Type:      r.Type,
		CreatedAt: r.CreatedAt,
	}

	if r.Properties.Valid && r.Properties.String != "" {
		var props map[string]interface{}
		if err := json.Unmarshal([]byte(r.Properties.String), &props); err != nil {
			return nil, err
		}
		rel.Properties = props
	}

	return rel, nil
}
