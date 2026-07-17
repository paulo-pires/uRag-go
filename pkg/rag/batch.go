package rag

import (
	"context"
	"fmt"
	"sync"
	"time"
)

type BatchResult struct {
	JobID string
	DocID string
	Error error
}

type JobStatus struct {
	JobID     string
	Status    string // "pending", "running", "completed", "failed"
	Total     int
	Completed int
	Failed    int
	Errors    []string
	CreatedAt time.Time
	UpdatedAt time.Time
}

type jobItem struct {
	jobID string
	doc   Document
}

type BatchProcessor struct {
	workers int
	queue   chan jobItem
	results chan BatchResult
	jobs    map[string]*JobStatus
	mu      sync.RWMutex
}

// NewBatchProcessor cria um novo processador de lote concorrente.
func NewBatchProcessor(workers int, queueSize int) *BatchProcessor {
	if workers <= 0 {
		workers = 2 // fallback razoável
	}
	return &BatchProcessor{
		workers: workers,
		queue:   make(chan jobItem, queueSize),
		results: make(chan BatchResult, queueSize*2),
		jobs:    make(map[string]*JobStatus),
	}
}

// Start inicializa os workers de ingestão concorrente.
func (bp *BatchProcessor) Start(ctx context.Context, store *UnifiedRAG) {
	// Lança os workers
	for i := 0; i < bp.workers; i++ {
		go bp.worker(ctx, store)
	}

	// Lança o controlador de resultados
	go bp.resultController(ctx)
}

// Submit submete uma lista de documentos para processamento assíncrono.
func (bp *BatchProcessor) Submit(docs []Document) (string, error) {
	if len(docs) == 0 {
		return "", fmt.Errorf("batch: lista de documentos vazia")
	}

	bp.mu.Lock()
	jobID := fmt.Sprintf("job_%d", time.Now().UnixNano())
	status := &JobStatus{
		JobID:     jobID,
		Status:    "pending",
		Total:     len(docs),
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	bp.jobs[jobID] = status
	bp.mu.Unlock()

	// Enfileira os documentos em background
	go func() {
		bp.mu.Lock()
		status.Status = "running"
		status.UpdatedAt = time.Now()
		bp.mu.Unlock()

		for _, doc := range docs {
			bp.queue <- jobItem{
				jobID: jobID,
				doc:   doc,
			}
		}
	}()

	return jobID, nil
}

// GetStatus retorna o status atualizado de um Job.
func (bp *BatchProcessor) GetStatus(jobID string) (*JobStatus, error) {
	bp.mu.RLock()
	defer bp.mu.RUnlock()

	status, ok := bp.jobs[jobID]
	if !ok {
		return nil, fmt.Errorf("batch: job com id %s não encontrado", jobID)
	}

	// Cria uma cópia para evitar data races em quem lê a struct
	copyStatus := *status
	copyStatus.Errors = append([]string(nil), status.Errors...)
	return &copyStatus, nil
}

func (bp *BatchProcessor) worker(ctx context.Context, store *UnifiedRAG) {
	for {
		select {
		case <-ctx.Done():
			return
		case item, ok := <-bp.queue:
			if !ok {
				return
			}

			// Ingestão no Vector Store (AddDocuments aceita fatia)
			err := store.AddDocuments(ctx, []Document{item.doc})

			bp.results <- BatchResult{
				JobID: item.jobID,
				DocID: item.doc.ID,
				Error: err,
			}
		}
	}
}

func (bp *BatchProcessor) resultController(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case res, ok := <-bp.results:
			if !ok {
				return
			}

			bp.mu.Lock()
			status, exists := bp.jobs[res.JobID]
			if exists {
				if res.Error != nil {
					status.Failed++
					status.Errors = append(status.Errors, fmt.Sprintf("doc %s: %v", res.DocID, res.Error))
				} else {
					status.Completed++
				}
				status.UpdatedAt = time.Now()

				// Verifica término do Job
				if status.Completed+status.Failed >= status.Total {
					if status.Failed == status.Total {
						status.Status = "failed"
					} else {
						status.Status = "completed"
					}
				}
			}
			bp.mu.Unlock()
		}
	}
}
