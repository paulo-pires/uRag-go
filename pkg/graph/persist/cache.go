package persist

import (
	"sync"
	"time"
)

// GraphCache cache em memória para operações rápidas
type GraphCache struct {
	entities  map[string]*Entity
	relations map[string][]*Relation // cache por chave composta
	mu        sync.RWMutex
	maxSize   int
	ttl       time.Duration
	created   map[string]time.Time
}

// NewGraphCache cria um novo cache
func NewGraphCache(maxSize int, ttl time.Duration) *GraphCache {
	if maxSize == 0 {
		maxSize = 1000 // default
	}
	if ttl == 0 {
		ttl = 5 * time.Minute // default
	}

	return &GraphCache{
		entities:  make(map[string]*Entity),
		relations: make(map[string][]*Relation),
		maxSize:   maxSize,
		ttl:       ttl,
		created:   make(map[string]time.Time),
	}
}

// SetEntity adiciona uma entidade ao cache
func (c *GraphCache) SetEntity(id string, entity *Entity) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Verifica se atingiu o limite
	if len(c.entities) >= c.maxSize {
		c.evictOldest()
	}

	c.entities[id] = entity
	c.created["entity:"+id] = time.Now()
}

// GetEntity busca uma entidade no cache
func (c *GraphCache) GetEntity(id string) *Entity {
	c.mu.RLock()
	defer c.mu.RUnlock()

	entity, ok := c.entities[id]
	if !ok {
		return nil
	}

	// Verifica TTL
	if created, ok := c.created["entity:"+id]; ok {
		if time.Since(created) > c.ttl {
			// Expirado, remover (mas não removeremos aqui para evitar deadlock)
			return nil
		}
	}

	return entity
}

// SetRelations adiciona relações ao cache
func (c *GraphCache) SetRelations(key string, relations []*Relation) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.relations[key] = relations
	c.created["rel:"+key] = time.Now()
}

// GetRelations busca relações no cache
func (c *GraphCache) GetRelations(key string) []*Relation {
	c.mu.RLock()
	defer c.mu.RUnlock()

	relations, ok := c.relations[key]
	if !ok {
		return nil
	}

	// Verifica TTL
	if created, ok := c.created["rel:"+key]; ok {
		if time.Since(created) > c.ttl {
			return nil
		}
	}

	return relations
}

// evictOldest remove o item mais antigo do cache
func (c *GraphCache) evictOldest() {
	var oldest string
	var oldestTime time.Time
	first := true

	for key, created := range c.created {
		if first || created.Before(oldestTime) {
			oldest = key
			oldestTime = created
			first = false
		}
	}

	if oldest != "" {
		delete(c.created, oldest)
		if len(oldest) > 7 && oldest[:7] == "entity:" {
			delete(c.entities, oldest[7:])
		} else if len(oldest) > 4 && oldest[:4] == "rel:" {
			delete(c.relations, oldest[4:])
		}
	}
}

// Clear limpa todo o cache
func (c *GraphCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.entities = make(map[string]*Entity)
	c.relations = make(map[string][]*Relation)
	c.created = make(map[string]time.Time)
}
