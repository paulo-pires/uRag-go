package rollout

import (
	"time"
)

// TraceRecord representa um registro histórico de execução de prompt/LLM em produção.
type TraceRecord struct {
	TraceID           string            `json:"trace_id" jsonschema:"identificador único do trace"`
	PromptID          string            `json:"prompt_id" jsonschema:"identificador do prompt"`
	Input             string            `json:"input" jsonschema:"input ou pergunta original"`
	OriginalPrompt    string            `json:"original_prompt" jsonschema:"prompt completo utilizado na execução original"`
	OriginalOutput    string            `json:"original_output" jsonschema:"resposta gerada originalmente"`
	OriginalScore     float64           `json:"original_score,omitempty" jsonschema:"score de avaliação original (0.0 a 1.0)"`
	OriginalLatencyMs float64           `json:"original_latency_ms,omitempty" jsonschema:"latência original em milissegundos"`
	OriginalTokens    int               `json:"original_tokens,omitempty" jsonschema:"contagem de tokens original"`
	Meta              map[string]string `json:"meta,omitempty" jsonschema:"metadados contextuais (tenant_id, user_id, etc.)"`
}

// ReplayConfig configura a execução do replay de traces.
type ReplayConfig struct {
	CandidatePrompt   string  `json:"candidate_prompt" jsonschema:"template ou conteúdo do prompt candidato"`
	CandidateModel    string  `json:"candidate_model,omitempty" jsonschema:"modelo a ser utilizado no replay"`
	CandidateProvider string  `json:"candidate_provider,omitempty" jsonschema:"provider (ollama ou openai)"`
	LLMBaseURL        string  `json:"llm_base_url,omitempty" jsonschema:"endpoint do provider de LLM"`
	LLMAPIKey         string  `json:"llm_api_key,omitempty" jsonschema:"API key para o provider de LLM"`
	MinFidelity       float64 `json:"min_fidelity,omitempty" jsonschema:"score mínimo de fidelidade exigido (0.0 a 1.0)"`
	Concurrency       int     `json:"concurrency,omitempty" jsonschema:"número de execuções paralelas (default 4)"`
	JudgeModel        string  `json:"judge_model,omitempty" jsonschema:"modelo para LLM-as-judge de fidelidade"`
}

// ReplayResult representa o resultado comparativo do replay de um trace individual.
type ReplayResult struct {
	TraceID            string  `json:"trace_id"`
	CandidateOutput    string  `json:"candidate_output"`
	CandidateLatencyMs float64 `json:"candidate_latency_ms"`
	CandidateTokens    int     `json:"candidate_tokens"`
	CandidateScore     float64 `json:"candidate_score"`
	FidelityScore      float64 `json:"fidelity_score"` // 0.0 a 1.0 (similaridade semântica/judge com output original)
	DeltaLatencyMs     float64 `json:"delta_latency_ms"`
	DeltaTokens        int     `json:"delta_tokens"`
	DeltaScore         float64 `json:"delta_score"`
	Passed             bool    `json:"passed"`
	Error              string  `json:"error,omitempty"`
}

// ReplayBatchSummary consolida as métricas agregadas e deltas do lote de replay.
type ReplayBatchSummary struct {
	TotalTraces          int            `json:"total_traces"`
	SuccessfulReplays    int            `json:"successful_replays"`
	FailedReplays        int            `json:"failed_replays"`
	AvgOriginalLatencyMs float64        `json:"avg_original_latency_ms"`
	AvgCandidateLatencyMs float64       `json:"avg_candidate_latency_ms"`
	DeltaLatencyMs       float64        `json:"delta_latency_ms"`
	AvgOriginalTokens    float64        `json:"avg_original_tokens"`
	AvgCandidateTokens   float64        `json:"avg_candidate_tokens"`
	DeltaTokens          float64        `json:"delta_tokens"`
	AvgFidelity          float64        `json:"avg_fidelity"`
	AvgOriginalScore     float64        `json:"avg_original_score"`
	AvgCandidateScore    float64        `json:"avg_candidate_score"`
	DeltaScore           float64        `json:"delta_score"`
	Results              []ReplayResult `json:"results"`
}

// RolloutConfig configura a estratégia de rollout gradual para um prompt.
type RolloutConfig struct {
	PromptID             string  `json:"prompt_id" jsonschema:"identificador único do prompt"`
	Strategy             string  `json:"strategy" jsonschema:"estratégia: canary, shadow ou ab_test"`
	BasePrompt           string  `json:"base_prompt" jsonschema:"versão base/estável atual do prompt"`
	CandidatePrompt      string  `json:"candidate_prompt" jsonschema:"versão candidata do prompt"`
	TrafficPercentage    float64 `json:"traffic_percentage" jsonschema:"porcentagem de tráfego para a versão candidata (0.0 a 100.0)"`
	AutoRollback         bool    `json:"auto_rollback" jsonschema:"ativa reversão automática caso métricas degradem"`
	MinQualityScore      float64 `json:"min_quality_score,omitempty" jsonschema:"score mínimo aceitável (default 0.70)"`
	MaxErrorRate         float64 `json:"max_error_rate,omitempty" jsonschema:"taxa máxima aceitável de erros (0.0 a 1.0, default 0.05)"`
	MaxLatencyMs         float64 `json:"max_latency_ms,omitempty" jsonschema:"latência máxima tolerada em ms"`
	DegradationThreshold float64 `json:"degradation_threshold,omitempty" jsonschema:"queda máxima tolerada no score em relação à base (default 0.15)"`
	MinSampleSize        int     `json:"min_sample_size,omitempty" jsonschema:"mínimo de requisições antes de avaliar auto-rollback (default 5)"`
}

// VariantMetrics mantém métricas de runtime para uma variante do prompt.
type VariantMetrics struct {
	Requests       int64   `json:"requests"`
	Errors         int64   `json:"errors"`
	TotalLatencyMs float64 `json:"total_latency_ms"`
	AvgLatencyMs   float64 `json:"avg_latency_ms"`
	TotalScore     float64 `json:"total_score"`
	AvgScore       float64 `json:"avg_score"`
	TotalTokens    int64   `json:"total_tokens"`
	AvgTokens      float64 `json:"avg_tokens"`
	ErrorRate      float64 `json:"error_rate"`
}

// RollbackEvent registra um evento de rollback manual ou automático.
type RollbackEvent struct {
	Timestamp      time.Time `json:"timestamp"`
	Reason         string    `json:"reason"`
	TriggerMetric  string    `json:"trigger_metric,omitempty"`
	TriggerValue   float64   `json:"trigger_value,omitempty"`
	ThresholdValue float64   `json:"threshold_value,omitempty"`
}

// RolloutState reflete o estado em tempo real de um rollout.
type RolloutState struct {
	PromptID        string                     `json:"prompt_id"`
	Config          RolloutConfig              `json:"config"`
	Status          string                     `json:"status"` // "active", "rolled_back", "completed", "paused"
	CreatedAt       time.Time                  `json:"created_at"`
	UpdatedAt       time.Time                  `json:"updated_at"`
	Metrics         map[string]*VariantMetrics `json:"metrics"`
	RollbackHistory []RollbackEvent            `json:"rollback_history"`
}

// RolloutDecision representa a resolução determinística de variante para uma requisição.
type RolloutDecision struct {
	PromptID        string `json:"prompt_id"`
	SelectedVariant string `json:"selected_variant"` // "base", "candidate", "shadow"
	Prompt          string `json:"prompt"`
	IsShadow        bool   `json:"is_shadow"`
	StickyKey       string `json:"sticky_key"`
	Bucket          int    `json:"bucket"`
	Reason          string `json:"reason"`
}
