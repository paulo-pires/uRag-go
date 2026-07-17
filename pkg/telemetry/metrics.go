package telemetry

import (
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type Collector struct {
	QueryCount  atomic.Uint64
	ErrorCount  atomic.Uint64
	CacheHits   atomic.Uint64
	CacheMisses atomic.Uint64

	mu              sync.RWMutex
	routerDecisions map[string]int
	storeLatencies  map[string][]time.Duration
}

// GlobalCollector é a instância padrão e global de telemetria do uRag-go.
var GlobalCollector = NewCollector()

// NewCollector cria e inicializa uma nova instância de Collector.
func NewCollector() *Collector {
	return &Collector{
		routerDecisions: make(map[string]int),
		storeLatencies:  make(map[string][]time.Duration),
	}
}

func (c *Collector) RecordQuery() {
	c.QueryCount.Add(1)
}

func (c *Collector) RecordError() {
	c.ErrorCount.Add(1)
}

func (c *Collector) RecordCacheHit() {
	c.CacheHits.Add(1)
}

func (c *Collector) RecordCacheMiss() {
	c.CacheMisses.Add(1)
}

func (c *Collector) RecordRouterDecision(strategy string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.routerDecisions[strategy]++
}

func (c *Collector) RecordLatency(store string, duration time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.storeLatencies[store] = append(c.storeLatencies[store], duration)
}

func (c *Collector) Reset() {
	c.QueryCount.Store(0)
	c.ErrorCount.Store(0)
	c.CacheHits.Store(0)
	c.CacheMisses.Store(0)

	c.mu.Lock()
	defer c.mu.Unlock()
	c.routerDecisions = make(map[string]int)
	c.storeLatencies = make(map[string][]time.Duration)
}

// RenderPrometheus formata as métricas coletadas no padrão texto Prometheus.
func (c *Collector) RenderPrometheus() string {
	var sb strings.Builder

	// Métricas básicas de queries e erros
	sb.WriteString("# HELP urag_queries_total Total number of queries routed.\n")
	sb.WriteString("# TYPE urag_queries_total counter\n")
	fmt.Fprintf(&sb, "urag_queries_total %d\n\n", c.QueryCount.Load())

	sb.WriteString("# HELP urag_errors_total Total number of query execution errors.\n")
	sb.WriteString("# TYPE urag_errors_total counter\n")
	fmt.Fprintf(&sb, "urag_errors_total %d\n\n", c.ErrorCount.Load())

	// Métricas de Cache de Embeddings
	sb.WriteString("# HELP urag_cache_hits_total Total number of embedding cache hits.\n")
	sb.WriteString("# TYPE urag_cache_hits_total counter\n")
	fmt.Fprintf(&sb, "urag_cache_hits_total %d\n\n", c.CacheHits.Load())

	sb.WriteString("# HELP urag_cache_misses_total Total number of embedding cache misses.\n")
	sb.WriteString("# TYPE urag_cache_misses_total counter\n")
	fmt.Fprintf(&sb, "urag_cache_misses_total %d\n\n", c.CacheMisses.Load())

	// Decisões do Roteador (Router Decisions) com labels
	sb.WriteString("# HELP urag_router_decision_total Router decisions count by strategy label.\n")
	sb.WriteString("# TYPE urag_router_decision_total counter\n")
	c.mu.RLock()
	for strategy, count := range c.routerDecisions {
		fmt.Fprintf(&sb, "urag_router_decision_total{strategy=\"%s\"} %d\n", strategy, count)
	}
	c.mu.RUnlock()
	sb.WriteString("\n")

	// Latências médias por Store
	sb.WriteString("# HELP urag_store_latency_average_seconds Average latency in seconds for each store.\n")
	sb.WriteString("# TYPE urag_store_latency_average_seconds gauge\n")
	c.mu.RLock()
	for store, list := range c.storeLatencies {
		if len(list) == 0 {
			continue
		}
		var total time.Duration
		for _, d := range list {
			total += d
		}
		avgSeconds := float64(total/time.Duration(len(list))) / float64(time.Second)
		fmt.Fprintf(&sb, "urag_store_latency_average_seconds{store=\"%s\"} %.6f\n", store, avgSeconds)
	}
	c.mu.RUnlock()

	return sb.String()
}
