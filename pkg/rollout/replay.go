package rollout

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"

	"urag-go/internal/ollama"
	"urag-go/internal/openai"
)

// LLMRunner define a função executora de chamadas ao modelo de linguagem.
type LLMRunner func(ctx context.Context, provider, baseURL, apiKey, model, prompt string) (string, error)

// JudgeRunner avalia a fidelidade e qualidade da resposta candidata em relação à original.
type JudgeRunner func(ctx context.Context, input, originalOutput, candidateOutput string) (fidelityScore float64, candidateScore float64, err error)

// ReplayEngine executa replay de traces históricos contra versões candidatas de prompt.
type ReplayEngine struct {
	llmRunner   LLMRunner
	judgeRunner JudgeRunner
}

// NewReplayEngine cria uma nova instância de ReplayEngine com executores injetados.
func NewReplayEngine(runner LLMRunner, judge JudgeRunner) *ReplayEngine {
	if runner == nil {
		runner = defaultLLMRunner
	}
	if judge == nil {
		judge = defaultJudgeRunner
	}
	return &ReplayEngine{
		llmRunner:   runner,
		judgeRunner: judge,
	}
}

// NewDefaultReplayEngine cria uma instância com os executores padrão do uRag-go.
func NewDefaultReplayEngine() *ReplayEngine {
	return NewReplayEngine(nil, nil)
}

// Replay processa um lote de traces históricos contra a configuração candidata.
func (e *ReplayEngine) Replay(ctx context.Context, traces []TraceRecord, cfg ReplayConfig) (ReplayBatchSummary, error) {
	if len(traces) == 0 {
		return ReplayBatchSummary{}, fmt.Errorf("lista de traces está vazia")
	}
	if strings.TrimSpace(cfg.CandidatePrompt) == "" {
		return ReplayBatchSummary{}, fmt.Errorf("candidate_prompt é obrigatório")
	}

	concurrency := cfg.Concurrency
	if concurrency <= 0 {
		concurrency = 4
	}
	if concurrency > len(traces) {
		concurrency = len(traces)
	}

	provider := cfg.CandidateProvider
	if provider == "" {
		provider = "ollama"
	}
	model := cfg.CandidateModel
	if model == "" {
		model = "granite4:micro-h"
	}

	results := make([]ReplayResult, len(traces))
	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup

	for i, trace := range traces {
		wg.Add(1)
		go func(idx int, tr TraceRecord) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			res := e.replaySingle(ctx, tr, cfg, provider, model)
			results[idx] = res
		}(i, trace)
	}

	wg.Wait()

	summary := aggregateSummary(traces, results)
	return summary, nil
}

func (e *ReplayEngine) replaySingle(ctx context.Context, tr TraceRecord, cfg ReplayConfig, provider, model string) ReplayResult {
	res := ReplayResult{
		TraceID: tr.TraceID,
	}

	prompt := renderPrompt(cfg.CandidatePrompt, tr.Input)
	start := time.Now()
	candidateOutput, err := e.llmRunner(ctx, provider, cfg.LLMBaseURL, cfg.LLMAPIKey, model, prompt)
	elapsedMs := float64(time.Since(start).Microseconds()) / 1000.0

	res.CandidateLatencyMs = elapsedMs
	res.DeltaLatencyMs = res.CandidateLatencyMs - tr.OriginalLatencyMs

	if err != nil {
		res.Error = err.Error()
		res.Passed = false
		return res
	}

	res.CandidateOutput = candidateOutput
	res.CandidateTokens = estimateTokens(candidateOutput)
	res.DeltaTokens = res.CandidateTokens - tr.OriginalTokens

	fidelity, candScore, judgeErr := e.judgeRunner(ctx, tr.Input, tr.OriginalOutput, candidateOutput)
	if judgeErr != nil {
		fidelity = computeLexicalFidelity(tr.OriginalOutput, candidateOutput)
		candScore = fidelity
	}

	res.FidelityScore = fidelity
	res.CandidateScore = candScore
	res.DeltaScore = res.CandidateScore - tr.OriginalScore

	passed := true
	if cfg.MinFidelity > 0 && res.FidelityScore < cfg.MinFidelity {
		passed = false
	}
	res.Passed = passed

	return res
}

func renderPrompt(template, input string) string {
	if strings.Contains(template, "{{input}}") {
		return strings.ReplaceAll(template, "{{input}}", input)
	}
	if strings.Contains(template, "{input}") {
		return strings.ReplaceAll(template, "{input}", input)
	}
	return template + "\n\n" + input
}

func estimateTokens(text string) int {
	words := len(strings.Fields(text))
	if words == 0 {
		return 0
	}
	return int(math.Ceil(float64(words) * 1.3))
}

func defaultLLMRunner(ctx context.Context, provider, baseURL, apiKey, model, prompt string) (string, error) {
	if provider == "openai" {
		return openai.Complete(ctx, baseURL, apiKey, model, prompt, false)
	}
	return ollama.Complete(ctx, baseURL, model, prompt, false)
}

func defaultJudgeRunner(_ context.Context, _, originalOutput, candidateOutput string) (float64, float64, error) {
	fidelity := computeLexicalFidelity(originalOutput, candidateOutput)
	return fidelity, fidelity, nil
}

func computeLexicalFidelity(orig, cand string) float64 {
	origWords := tokenize(orig)
	candWords := tokenize(cand)

	if len(origWords) == 0 && len(candWords) == 0 {
		return 1.0
	}
	if len(origWords) == 0 || len(candWords) == 0 {
		return 0.0
	}

	origSet := make(map[string]struct{}, len(origWords))
	for _, w := range origWords {
		origSet[w] = struct{}{}
	}

	intersection := 0
	candSet := make(map[string]struct{}, len(candWords))
	for _, w := range candWords {
		candSet[w] = struct{}{}
		if _, ok := origSet[w]; ok {
			intersection++
		}
	}

	union := len(origSet)
	for w := range candSet {
		if _, ok := origSet[w]; !ok {
			union++
		}
	}

	if union == 0 {
		return 1.0
	}
	return float64(intersection) / float64(union)
}

func tokenize(text string) []string {
	clean := strings.ToLower(text)
	var words []string
	for _, w := range strings.Fields(clean) {
		w = strings.Trim(w, ".,!?:;\"'()[]{}<>-")
		if w != "" {
			words = append(words, w)
		}
	}
	return words
}

func aggregateSummary(traces []TraceRecord, results []ReplayResult) ReplayBatchSummary {
	summary := ReplayBatchSummary{
		TotalTraces: len(results),
		Results:     results,
	}

	var sumOrigLat, sumCandLat, sumFidelity, sumOrigScore, sumCandScore float64
	var sumOrigTok, sumCandTok int64

	for i, r := range results {
		if r.Error == "" {
			summary.SuccessfulReplays++
		} else {
			summary.FailedReplays++
		}

		tr := traces[i]
		sumOrigLat += tr.OriginalLatencyMs
		sumCandLat += r.CandidateLatencyMs
		sumOrigTok += int64(tr.OriginalTokens)
		sumCandTok += int64(r.CandidateTokens)
		sumFidelity += r.FidelityScore
		sumOrigScore += tr.OriginalScore
		sumCandScore += r.CandidateScore
	}

	n := float64(len(results))
	if n > 0 {
		summary.AvgOriginalLatencyMs = sumOrigLat / n
		summary.AvgCandidateLatencyMs = sumCandLat / n
		summary.DeltaLatencyMs = summary.AvgCandidateLatencyMs - summary.AvgOriginalLatencyMs

		summary.AvgOriginalTokens = float64(sumOrigTok) / n
		summary.AvgCandidateTokens = float64(sumCandTok) / n
		summary.DeltaTokens = summary.AvgCandidateTokens - summary.AvgOriginalTokens

		summary.AvgFidelity = sumFidelity / n
		summary.AvgOriginalScore = sumOrigScore / n
		summary.AvgCandidateScore = sumCandScore / n
		summary.DeltaScore = summary.AvgCandidateScore - summary.AvgOriginalScore
	}

	return summary
}
