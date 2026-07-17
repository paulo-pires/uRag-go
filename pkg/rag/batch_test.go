package rag

import (
	"context"
	"fmt"
	"math"
	"testing"
	"time"
)

func TestBatchProcessorMultipleWorkers(t *testing.T) {
	// 1. Inicia UnifiedRAG com mock de embedding
	store, err := NewWithEmbedding(Config{}, func(ctx context.Context, text string) ([]float32, error) {
		seed := len(text)
		vec := make([]float32, 1536)
		var sumSq float64
		for i := range vec {
			val := float32((seed >> uint(i%8) & 0xFF)) / 255
			vec[i] = val
			sumSq += float64(val * val)
		}
		norm := float32(math.Sqrt(sumSq))
		for i := range vec {
			vec[i] /= norm
		}
		return vec, nil
	})
	if err != nil {
		t.Fatalf("falha ao iniciar vector store de testes: %v", err)
	}
	defer store.Close()

	// 2. Inicia o BatchProcessor
	bp := NewBatchProcessor(3, 10)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	bp.Start(ctx, store)

	// 3. Submete 5 documentos de teste
	var docs []Document
	for i := 1; i <= 5; i++ {
		docs = append(docs, Document{
			ID:      fmt.Sprintf("doc_batch_%d", i),
			Content: fmt.Sprintf("Conteúdo do documento de teste em lote %d", i),
			Source:  "batch_test",
		})
	}

	jobID, err := bp.Submit(docs)
	if err != nil {
		t.Fatalf("falha ao submeter job: %v", err)
	}

	// 4. Aguarda processamento terminar
	var status *JobStatus
	for i := 0; i < 20; i++ {
		status, err = bp.GetStatus(jobID)
		if err != nil {
			t.Fatalf("falha ao pegar status: %v", err)
		}
		if status.Status == "completed" || status.Status == "failed" {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}

	// 5. Validações finais
	if status.Status != "completed" {
		t.Errorf("esperava status completed, obtido: %s (com %d falhas)", status.Status, status.Failed)
	}

	if status.Completed != 5 {
		t.Errorf("esperava 5 documentos completados, obtido %d", status.Completed)
	}

	// Valida se os documentos realmente entraram no vector store
	res, err := store.Query(context.Background(), "documento de teste", 5)
	if err != nil {
		t.Fatalf("erro ao fazer query no store: %v", err)
	}

	if len(res) != 5 {
		t.Errorf("esperava recuperar 5 documentos, obtido %d", len(res))
	}
}
