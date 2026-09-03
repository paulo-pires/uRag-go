package rollout

import (
	"fmt"
	"hash/fnv"
	"sync"
	"time"
)

// RolloutManager gerencia estratégias de rollout gradual, decisões determinísticas de tráfego
// e guardrails de auto-rollback para variantes de prompts.
type RolloutManager struct {
	mu     sync.RWMutex
	states map[string]*RolloutState
}

// NewManager cria uma nova instância de RolloutManager.
func NewManager() *RolloutManager {
	return &RolloutManager{
		states: make(map[string]*RolloutState),
	}
}

// ConfigureRollout inicializa ou atualiza a estratégia de rollout para um prompt.
func (m *RolloutManager) ConfigureRollout(cfg RolloutConfig) (*RolloutState, error) {
	if cfg.PromptID == "" {
		return nil, fmt.Errorf("prompt_id é obrigatório")
	}
	if cfg.BasePrompt == "" {
		return nil, fmt.Errorf("base_prompt é obrigatório")
	}

	strategy := cfg.Strategy
	if strategy == "" {
		strategy = "canary"
	}
	switch strategy {
	case "canary", "shadow", "ab_test":
	default:
		return nil, fmt.Errorf("estratégia inválida: %q (esperado canary, shadow ou ab_test)", strategy)
	}
	cfg.Strategy = strategy

	if cfg.TrafficPercentage < 0 {
		cfg.TrafficPercentage = 0
	} else if cfg.TrafficPercentage > 100 {
		cfg.TrafficPercentage = 100
	}

	if cfg.MinQualityScore <= 0 {
		cfg.MinQualityScore = 0.70
	}
	if cfg.MaxErrorRate <= 0 {
		cfg.MaxErrorRate = 0.05
	}
	if cfg.DegradationThreshold <= 0 {
		cfg.DegradationThreshold = 0.15
	}
	if cfg.MinSampleSize <= 0 {
		cfg.MinSampleSize = 5
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	now := time.Now().UTC()
	state, exists := m.states[cfg.PromptID]
	if !exists {
		state = &RolloutState{
			PromptID:  cfg.PromptID,
			CreatedAt: now,
			Metrics: map[string]*VariantMetrics{
				"base":      {},
				"candidate": {},
			},
			RollbackHistory: make([]RollbackEvent, 0),
		}
		m.states[cfg.PromptID] = state
	}

	state.Config = cfg
	state.Status = "active"
	state.UpdatedAt = now

	return state.clone(), nil
}

// GetStatus retorna o estado atual e métricas de rollout de um prompt.
func (m *RolloutManager) GetStatus(promptID string) (*RolloutState, error) {
	if promptID == "" {
		return nil, fmt.Errorf("prompt_id é obrigatório")
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	state, ok := m.states[promptID]
	if !ok {
		return nil, fmt.Errorf("rollout para prompt_id %q não encontrado", promptID)
	}

	return state.clone(), nil
}

// EvaluateDecision resolve determinísticamente qual variante do prompt deve ser executada.
func (m *RolloutManager) EvaluateDecision(promptID string, stickyKey string) (RolloutDecision, error) {
	if promptID == "" {
		return RolloutDecision{}, fmt.Errorf("prompt_id é obrigatório")
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	state, ok := m.states[promptID]
	if !ok {
		return RolloutDecision{}, fmt.Errorf("rollout para prompt_id %q não encontrado", promptID)
	}

	decision := RolloutDecision{
		PromptID:  promptID,
		StickyKey: stickyKey,
	}

	if state.Status == "rolled_back" {
		decision.SelectedVariant = "base"
		decision.Prompt = state.Config.BasePrompt
		decision.Reason = "rollout reverted (status: rolled_back)"
		return decision, nil
	}

	if state.Status == "completed" {
		decision.SelectedVariant = "candidate"
		decision.Prompt = state.Config.CandidatePrompt
		decision.Reason = "rollout 100% completed"
		return decision, nil
	}

	if state.Status == "paused" {
		decision.SelectedVariant = "base"
		decision.Prompt = state.Config.BasePrompt
		decision.Reason = "rollout paused"
		return decision, nil
	}

	if state.Config.Strategy == "shadow" {
		decision.SelectedVariant = "base"
		decision.Prompt = state.Config.BasePrompt
		decision.IsShadow = true
		decision.Reason = "shadow mode active (execute base, mirror to candidate)"
		return decision, nil
	}

	// Estratégias 'canary' e 'ab_test': hashing consistente pelo stickyKey + promptID
	bucket := computeBucket(stickyKey, promptID)
	decision.Bucket = bucket

	if float64(bucket) < state.Config.TrafficPercentage {
		decision.SelectedVariant = "candidate"
		decision.Prompt = state.Config.CandidatePrompt
		decision.Reason = fmt.Sprintf("deterministic bucket %d < %.1f%% (%s)", bucket, state.Config.TrafficPercentage, state.Config.Strategy)
	} else {
		decision.SelectedVariant = "base"
		decision.Prompt = state.Config.BasePrompt
		decision.Reason = fmt.Sprintf("deterministic bucket %d >= %.1f%% (%s)", bucket, state.Config.TrafficPercentage, state.Config.Strategy)
	}

	return decision, nil
}

// RecordMetric registra métricas de execução de uma variante e verifica guardrails de auto-rollback.
func (m *RolloutManager) RecordMetric(promptID string, variant string, latencyMs float64, score float64, tokens int, isError bool) (*RolloutState, bool, error) {
	if promptID == "" {
		return nil, false, fmt.Errorf("prompt_id é obrigatório")
	}
	if variant != "base" && variant != "candidate" && variant != "shadow" {
		return nil, false, fmt.Errorf("variante inválida: %q", variant)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	state, ok := m.states[promptID]
	if !ok {
		return nil, false, fmt.Errorf("rollout para prompt_id %q não encontrado", promptID)
	}

	metricKey := variant
	if variant == "shadow" {
		metricKey = "candidate"
	}

	vm, exists := state.Metrics[metricKey]
	if !exists || vm == nil {
		vm = &VariantMetrics{}
		state.Metrics[metricKey] = vm
	}

	vm.Requests++
	if isError {
		vm.Errors++
	}
	vm.TotalLatencyMs += latencyMs
	vm.AvgLatencyMs = vm.TotalLatencyMs / float64(vm.Requests)

	if score > 0 {
		vm.TotalScore += score
		vm.AvgScore = vm.TotalScore / float64(vm.Requests)
	}
	if tokens > 0 {
		vm.TotalTokens += int64(tokens)
		vm.AvgTokens = float64(vm.TotalTokens) / float64(vm.Requests)
	}
	vm.ErrorRate = float64(vm.Errors) / float64(vm.Requests)

	state.UpdatedAt = time.Now().UTC()

	// Checa Auto-Rollback se estiver ativo e a métrica for do candidate
	rolledBack := false
	if state.Config.AutoRollback && state.Status == "active" && metricKey == "candidate" {
		if int(vm.Requests) >= state.Config.MinSampleSize {
			trigger, metric, val, th := checkAutoRollbackTrigger(state)
			if trigger {
				event := RollbackEvent{
					Timestamp:      time.Now().UTC(),
					Reason:         fmt.Sprintf("auto-rollback: %s (value=%.4f, threshold=%.4f)", metric, val, th),
					TriggerMetric:  metric,
					TriggerValue:   val,
					ThresholdValue: th,
				}
				state.RollbackHistory = append(state.RollbackHistory, event)
				state.Status = "rolled_back"
				rolledBack = true
			}
		}
	}

	return state.clone(), rolledBack, nil
}

// Rollback reverte manualmente um rollout para a versão base.
func (m *RolloutManager) Rollback(promptID string, reason string) (*RolloutState, error) {
	if promptID == "" {
		return nil, fmt.Errorf("prompt_id é obrigatório")
	}
	if reason == "" {
		reason = "manual rollback"
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	state, ok := m.states[promptID]
	if !ok {
		return nil, fmt.Errorf("rollout para prompt_id %q não encontrado", promptID)
	}

	event := RollbackEvent{
		Timestamp: time.Now().UTC(),
		Reason:    reason,
	}
	state.RollbackHistory = append(state.RollbackHistory, event)
	state.Status = "rolled_back"
	state.UpdatedAt = time.Now().UTC()

	return state.clone(), nil
}

func checkAutoRollbackTrigger(state *RolloutState) (bool, string, float64, float64) {
	cand := state.Metrics["candidate"]
	base := state.Metrics["base"]
	cfg := state.Config

	// 1. Taxa de erro
	if cfg.MaxErrorRate > 0 && cand.ErrorRate > cfg.MaxErrorRate {
		return true, "error_rate", cand.ErrorRate, cfg.MaxErrorRate
	}

	// 2. Score de qualidade abaixo do mínimo
	if cfg.MinQualityScore > 0 && cand.AvgScore > 0 && cand.AvgScore < cfg.MinQualityScore {
		return true, "min_quality_score", cand.AvgScore, cfg.MinQualityScore
	}

	// 3. Degradação de qualidade em relação à base
	if cfg.DegradationThreshold > 0 && base != nil && base.Requests >= int64(cfg.MinSampleSize) && base.AvgScore > 0 && cand.AvgScore > 0 {
		drop := base.AvgScore - cand.AvgScore
		if drop > cfg.DegradationThreshold {
			return true, "quality_degradation", drop, cfg.DegradationThreshold
		}
	}

	// 4. Latência máxima
	if cfg.MaxLatencyMs > 0 && cand.AvgLatencyMs > cfg.MaxLatencyMs {
		return true, "max_latency_ms", cand.AvgLatencyMs, cfg.MaxLatencyMs
	}

	return false, "", 0, 0
}

func computeBucket(stickyKey, promptID string) int {
	h := fnv.New32a()
	if stickyKey == "" {
		h.Write([]byte(promptID))
	} else {
		h.Write([]byte(stickyKey + ":" + promptID))
	}
	return int(h.Sum32() % 100)
}

func (s *RolloutState) clone() *RolloutState {
	if s == nil {
		return nil
	}
	c := &RolloutState{
		PromptID:        s.PromptID,
		Config:          s.Config,
		Status:          s.Status,
		CreatedAt:       s.CreatedAt,
		UpdatedAt:       s.UpdatedAt,
		Metrics:         make(map[string]*VariantMetrics, len(s.Metrics)),
		RollbackHistory: make([]RollbackEvent, len(s.RollbackHistory)),
	}
	for k, v := range s.Metrics {
		if v != nil {
			metricCopy := *v
			c.Metrics[k] = &metricCopy
		}
	}
	copy(c.RollbackHistory, s.RollbackHistory)
	return c
}
