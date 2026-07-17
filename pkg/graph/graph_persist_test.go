package graph

import (
	"context"
	"os"
	"testing"
	"time"

	"urag-go/pkg/graph/persist"
	"urag-go/pkg/rag"
)

func TestGraphStorePersistIntegration(t *testing.T) {
	// Cria um arquivo temporário de banco para persistência
	dbFile := "test_graph_persist.db"
	defer os.Remove(dbFile)

	persistCfg := persist.Config{
		DSN:       dbFile,
		CacheSize: 10,
		CacheTTL:  1 * time.Minute,
	}

	// 1. Inicializa o GraphStore com persistência e a completion fake
	g, err := NewWithCompletionAndPersist(fakeExtraction, persistCfg)
	if err != nil {
		t.Fatalf("erro ao criar GraphStore: %v", err)
	}

	docs := []rag.Document{
		{ID: "doc1", Content: "Maria trabalha na empresa Ignus."},
		{ID: "doc2", Content: "Ignus fica no Brasil."},
	}

	ctx := context.Background()

	// 2. Adiciona documentos (isso deve rodar a extração e salvar no SQLite)
	if err := g.AddDocuments(ctx, docs); err != nil {
		t.Fatalf("AddDocuments falhou: %v", err)
	}

	// 3. Testa estatísticas locais
	stats := g.GetStats()
	if stats["persisted"] != true {
		t.Errorf("esperava persisted=true nas estatísticas, obtido: %v", stats["persisted"])
	}
	if stats["entities"] != 3 { // maria, ignus, brasil
		t.Errorf("esperava 3 entidades, obtido: %v", stats["entities"])
	}
	if stats["relations"] != 2 { // trabalha_em, localizada_em
		t.Errorf("esperava 2 relações, obtido: %v", stats["relations"])
	}

	// 4. Executa query
	results, err := g.Query(ctx, "onde Maria trabalha?", 2)
	if err != nil {
		t.Fatalf("Query falhou: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("esperava 2 relações na busca multi-hop, obtido: %d", len(results))
	}

	// Fecha o store para liberar o arquivo SQLite
	if err := g.Close(); err != nil {
		t.Fatalf("Close falhou: %v", err)
	}

	// 5. Instancia um novo GraphStore apontando para o mesmo banco para testar o loadFromStore
	g2, err := NewWithCompletionAndPersist(fakeExtraction, persistCfg)
	if err != nil {
		t.Fatalf("erro ao re-criar GraphStore: %v", err)
	}
	defer g2.Close()

	// Verifica se os dados foram recarregados corretamente na inicialização
	stats2 := g2.GetStats()
	if stats2["entities"] != 3 {
		t.Errorf("esperava 3 entidades carregadas, obtido: %v", stats2["entities"])
	}
	if stats2["relations"] != 2 {
		t.Errorf("esperava 2 relações carregadas, obtido: %v", stats2["relations"])
	}

	// 6. Testar LoadFullGraph
	snapshot, err := g2.LoadFullGraph(ctx)
	if err != nil {
		t.Fatalf("LoadFullGraph falhou: %v", err)
	}
	if len(snapshot.Entities) != 3 {
		t.Errorf("esperava 3 entidades no snapshot, obtido: %d", len(snapshot.Entities))
	}
	if len(snapshot.Relations) != 2 {
		t.Errorf("esperava 2 relações no snapshot, obtido: %d", len(snapshot.Relations))
	}
}
