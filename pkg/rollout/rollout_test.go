package rollout

import (
	"context"
	"fmt"
	"sync"
	"testing"
)

func TestRolloutManagerConfigureAndEvaluate(t *testing.T) {
	mgr := NewManager()

	cfg := RolloutConfig{
		PromptID:          "p-search-01",
		Strategy:          "canary",
		BasePrompt:        "Prompt Base v1",
		CandidatePrompt:   "Prompt Candidate v2",
		TrafficPercentage: 20.0,
		AutoRollback:      true,
	}

	state, err := mgr.ConfigureRollout(cfg)
	if err != nil {
		t.Fatalf("ConfigureRollout failed: %v", err)
	}

	if state.Status != "active" {
		t.Fatalf("expected status active, got %s", state.Status)
	}

	// Avaliação determinística para diferentes sticky keys
	d1, err := mgr.EvaluateDecision("p-search-01", "tenant-alpha")
	if err != nil {
		t.Fatalf("EvaluateDecision failed: %v", err)
	}
	d2, err := mgr.EvaluateDecision("p-search-01", "tenant-alpha")
	if err != nil {
		t.Fatalf("EvaluateDecision failed: %v", err)
	}

	if d1.SelectedVariant != d2.SelectedVariant {
		t.Fatalf("expected deterministic decision, got %s and %s", d1.SelectedVariant, d2.SelectedVariant)
	}
	if d1.Prompt == "" {
		t.Fatal("expected non-empty prompt")
	}
}

func TestRolloutManagerShadowMode(t *testing.T) {
	mgr := NewManager()

	cfg := RolloutConfig{
		PromptID:        "p-shadow-01",
		Strategy:        "shadow",
		BasePrompt:      "Base",
		CandidatePrompt: "Candidate",
	}

	_, err := mgr.ConfigureRollout(cfg)
	if err != nil {
		t.Fatalf("ConfigureRollout failed: %v", err)
	}

	d, err := mgr.EvaluateDecision("p-shadow-01", "user-123")
	if err != nil {
		t.Fatalf("EvaluateDecision failed: %v", err)
	}

	if !d.IsShadow {
		t.Fatal("expected IsShadow=true in shadow strategy")
	}
	if d.SelectedVariant != "base" {
		t.Fatalf("expected main variant to be base in shadow mode, got %s", d.SelectedVariant)
	}
}

func TestRolloutManagerAutoRollbackOnErrorRate(t *testing.T) {
	mgr := NewManager()

	cfg := RolloutConfig{
		PromptID:          "p-auto-rb",
		Strategy:          "canary",
		BasePrompt:        "Base",
		CandidatePrompt:   "Candidate",
		TrafficPercentage: 50.0,
		AutoRollback:      true,
		MaxErrorRate:      0.20,
		MinSampleSize:     5,
	}

	_, err := mgr.ConfigureRollout(cfg)
	if err != nil {
		t.Fatalf("ConfigureRollout failed: %v", err)
	}

	// Registra 5 requisições com 3 erros no candidate (taxa de erro 60% > 20%)
	for i := 0; i < 5; i++ {
		isErr := i >= 2
		st, rolledBack, err := mgr.RecordMetric("p-auto-rb", "candidate", 120.0, 0.85, 50, isErr)
		if err != nil {
			t.Fatalf("RecordMetric failed: %v", err)
		}
		if i == 4 {
			if !rolledBack {
				t.Fatal("expected auto-rollback on 5th sample exceeding max error rate")
			}
			if st.Status != "rolled_back" {
				t.Fatalf("expected status rolled_back, got %s", st.Status)
			}
			if len(st.RollbackHistory) == 0 {
				t.Fatal("expected rollback history event recorded")
			}
		}
	}

	// Decisões pós-rollback devem apontar para a base
	d, err := mgr.EvaluateDecision("p-auto-rb", "any-key")
	if err != nil {
		t.Fatalf("EvaluateDecision failed: %v", err)
	}
	if d.SelectedVariant != "base" {
		t.Fatalf("expected base variant after rollback, got %s", d.SelectedVariant)
	}
}

func TestRolloutManagerManualRollbackAndGetStatus(t *testing.T) {
	mgr := NewManager()

	cfg := RolloutConfig{
		PromptID:        "p-manual-rb",
		BasePrompt:      "Base",
		CandidatePrompt: "Candidate",
	}
	_, _ = mgr.ConfigureRollout(cfg)

	st, err := mgr.Rollback("p-manual-rb", "Degradação percebida por operador")
	if err != nil {
		t.Fatalf("Rollback failed: %v", err)
	}
	if st.Status != "rolled_back" {
		t.Fatalf("expected rolled_back, got %s", st.Status)
	}

	status, err := mgr.GetStatus("p-manual-rb")
	if err != nil {
		t.Fatalf("GetStatus failed: %v", err)
	}
	if status.Status != "rolled_back" {
		t.Fatalf("expected status rolled_back, got %s", status.Status)
	}
}

func TestRolloutManagerConcurrency(t *testing.T) {
	mgr := NewManager()

	cfg := RolloutConfig{
		PromptID:          "p-concurrent",
		BasePrompt:        "Base",
		CandidatePrompt:   "Candidate",
		TrafficPercentage: 30.0,
		AutoRollback:      false,
	}
	_, _ = mgr.ConfigureRollout(cfg)

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			key := fmt.Sprintf("tenant-%d", id)
			_, _ = mgr.EvaluateDecision("p-concurrent", key)
			_, _, _ = mgr.RecordMetric("p-concurrent", "candidate", 100.0, 0.9, 40, false)
			_, _ = mgr.GetStatus("p-concurrent")
		}(i)
	}
	wg.Wait()
}

func TestReplayEngine(t *testing.T) {
	fakeRunner := func(_ context.Context, _, _, _, _, prompt string) (string, error) {
		return "Resposta candidata refinada: " + prompt, nil
	}

	fakeJudge := func(_ context.Context, _, _, _ string) (float64, float64, error) {
		return 0.92, 0.95, nil
	}

	engine := NewReplayEngine(fakeRunner, fakeJudge)

	traces := []TraceRecord{
		{
			TraceID:           "tr-01",
			PromptID:          "prompt-1",
			Input:             "Qual o horário de atendimento?",
			OriginalPrompt:    "Prompt v1",
			OriginalOutput:    "Atendemos de 8h às 18h",
			OriginalScore:     0.80,
			OriginalLatencyMs: 250.0,
			OriginalTokens:    15,
		},
		{
			TraceID:           "tr-02",
			PromptID:          "prompt-1",
			Input:             "Vocês aceitam PIX?",
			OriginalPrompt:    "Prompt v1",
			OriginalOutput:    "Sim, aceitamos PIX e Cartão",
			OriginalScore:     0.85,
			OriginalLatencyMs: 220.0,
			OriginalTokens:    20,
		},
	}

	summary, err := engine.Replay(context.Background(), traces, ReplayConfig{
		CandidatePrompt: "Você é um assistente prestativo. Responda: {{input}}",
		MinFidelity:     0.80,
		Concurrency:     2,
	})

	if err != nil {
		t.Fatalf("Replay failed: %v", err)
	}

	if summary.TotalTraces != 2 {
		t.Fatalf("expected 2 traces, got %d", summary.TotalTraces)
	}
	if summary.SuccessfulReplays != 2 {
		t.Fatalf("expected 2 successful replays, got %d", summary.SuccessfulReplays)
	}
	if summary.AvgFidelity < 0.90 {
		t.Fatalf("expected high fidelity, got %f", summary.AvgFidelity)
	}
	if len(summary.Results) != 2 {
		t.Fatalf("expected 2 result items, got %d", len(summary.Results))
	}
	if !summary.Results[0].Passed {
		t.Fatal("expected trace 1 to pass fidelity threshold")
	}
}
