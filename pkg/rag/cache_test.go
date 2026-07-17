package rag

import (
	"testing"
	"time"
)

func TestEmbeddingCacheLRUBehavior(t *testing.T) {
	// Cria cache com tamanho máximo 2 e TTL longo
	cache := NewEmbeddingCache(2, 10*time.Second)

	v1 := []float32{1.0, 2.0}
	v2 := []float32{3.0, 4.0}
	v3 := []float32{5.0, 6.0}

	cache.Set("key1", v1)
	cache.Set("key2", v2)

	// Acessa key1 para torná-lo o mais recente
	if _, ok := cache.Get("key1"); !ok {
		t.Error("esperava key1 no cache")
	}

	// Adiciona terceiro item (deve desalojar o mais antigo, que agora é key2)
	cache.Set("key3", v3)

	if _, ok := cache.Get("key1"); !ok {
		t.Error("esperava key1 continuar no cache (acessado recentemente)")
	}
	if _, ok := cache.Get("key3"); !ok {
		t.Error("esperava key3 no cache")
	}
	if _, ok := cache.Get("key2"); ok {
		t.Error("esperava key2 ter sido desalojado pelo LRU")
	}
}

func TestEmbeddingCacheTTL(t *testing.T) {
	// Cria cache com TTL muito curto
	cache := NewEmbeddingCache(10, 50*time.Millisecond)

	cache.Set("key1", []float32{1.0})

	// Valida hit imediato
	if _, ok := cache.Get("key1"); !ok {
		t.Fatal("esperava key1 no cache")
	}

	// Espera expiração
	time.Sleep(60 * time.Millisecond)

	// Valida miss após TTL
	if _, ok := cache.Get("key1"); ok {
		t.Error("esperava key1 ter expirado pelo TTL")
	}
}

func TestEmbeddingCacheStats(t *testing.T) {
	cache := NewEmbeddingCache(5, 1*time.Minute)

	cache.Set("key1", []float32{1.0})

	// 1 Hit
	cache.Get("key1")

	// 1 Miss
	cache.Get("key_non_existent")

	hits, misses := cache.Stats()
	if hits != 1 {
		t.Errorf("esperava 1 hit, obtido: %d", hits)
	}
	if misses != 1 {
		t.Errorf("esperava 1 miss, obtido: %d", misses)
	}

	cache.Clear()
	hits, misses = cache.Stats()
	if hits != 0 || misses != 0 {
		t.Errorf("esperava stats zerados após Clear, obtido: hits=%d, misses=%d", hits, misses)
	}
}
