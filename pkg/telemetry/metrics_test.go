package telemetry

import (
	"strings"
	"testing"
	"time"
)

func TestTelemetryCollectorAndPrometheusRendering(t *testing.T) {
	c := NewCollector()

	// 1. Grava eventos nas métricas
	c.RecordQuery()
	c.RecordQuery()
	c.RecordError()

	c.RecordCacheHit()
	c.RecordCacheHit()
	c.RecordCacheMiss()

	c.RecordRouterDecision("vector")
	c.RecordRouterDecision("vector")
	c.RecordRouterDecision("graph")

	c.RecordLatency("vector", 100*time.Millisecond)
	c.RecordLatency("vector", 200*time.Millisecond)

	// 2. Renderiza no padrão Prometheus
	output := c.RenderPrometheus()

	// 3. Validações
	expectedSubstrings := []string{
		"urag_queries_total 2",
		"urag_errors_total 1",
		"urag_cache_hits_total 2",
		"urag_cache_misses_total 1",
		`urag_router_decision_total{strategy="vector"} 2`,
		`urag_router_decision_total{strategy="graph"} 1`,
		`urag_store_latency_average_seconds{store="vector"} 0.150000`, // média de 100ms e 200ms = 150ms = 0.15s
	}

	for _, sub := range expectedSubstrings {
		if !strings.Contains(output, sub) {
			t.Errorf("esperava encontrar %q no output Prometheus, obtido:\n%s", sub, output)
		}
	}

	// 4. Teste de Reset
	c.Reset()
	if c.QueryCount.Load() != 0 {
		t.Error("esperava QueryCount zerado após Reset")
	}
	if len(c.routerDecisions) != 0 {
		t.Error("esperava routerDecisions limpo após Reset")
	}
}
