package rag

import (
	"container/list"
	"sync"
	"time"

	"urag-go/pkg/telemetry"
)

type cacheEntry struct {
	key       string
	value     []float32
	createdAt time.Time
}

type EmbeddingCache struct {
	mu        sync.RWMutex
	maxSize   int
	ttl       time.Duration
	ll        *list.List
	cache     map[string]*list.Element
	hitCount  uint64
	missCount uint64
}

// NewEmbeddingCache cria um novo cache de embeddings
func NewEmbeddingCache(maxSize int, ttl time.Duration) *EmbeddingCache {
	if maxSize <= 0 {
		maxSize = 1000 // default
	}
	if ttl <= 0 {
		ttl = 5 * time.Minute // default
	}
	return &EmbeddingCache{
		maxSize: maxSize,
		ttl:     ttl,
		ll:      list.New(),
		cache:   make(map[string]*list.Element),
	}
}

// Get busca no cache
func (c *EmbeddingCache) Get(key string) ([]float32, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if ele, hit := c.cache[key]; hit {
		entry := ele.Value.(*cacheEntry)
		if time.Since(entry.createdAt) > c.ttl {
			// Expirado
			c.ll.Remove(ele)
			delete(c.cache, key)
			c.missCount++
			telemetry.GlobalCollector.RecordCacheMiss()
			return nil, false
		}
		c.ll.MoveToFront(ele)
		c.hitCount++
		telemetry.GlobalCollector.RecordCacheHit()
		return entry.value, true
	}

	c.missCount++
	telemetry.GlobalCollector.RecordCacheMiss()
	return nil, false
}

// Set adiciona ou atualiza no cache
func (c *EmbeddingCache) Set(key string, val []float32) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if ele, ok := c.cache[key]; ok {
		c.ll.MoveToFront(ele)
		entry := ele.Value.(*cacheEntry)
		entry.value = val
		entry.createdAt = time.Now()
		return
	}

	// Remove mais antigo se atingiu limite
	if c.ll.Len() >= c.maxSize {
		c.removeOldest()
	}

	entry := &cacheEntry{
		key:       key,
		value:     val,
		createdAt: time.Now(),
	}
	ele := c.ll.PushFront(entry)
	c.cache[key] = ele
}

func (c *EmbeddingCache) removeOldest() {
	ele := c.ll.Back()
	if ele != nil {
		c.ll.Remove(ele)
		entry := ele.Value.(*cacheEntry)
		delete(c.cache, entry.key)
	}
}

// Clear esvazia o cache
func (c *EmbeddingCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.ll = list.New()
	c.cache = make(map[string]*list.Element)
	c.hitCount = 0
	c.missCount = 0
}

// Stats retorna hits e misses
func (c *EmbeddingCache) Stats() (uint64, uint64) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.hitCount, c.missCount
}