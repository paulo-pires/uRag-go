package tree

import (
	"context"
	"strconv"
	"strings"
	"testing"
)

// Dois níveis reais de aninhamento, com vocabulário compartilhado entre
// capítulo e seção — necessário porque fakeNavigate só entende substring,
// não semântica (um "Animais" genérico sem a palavra "gatos" não seria achado).
const testDoc = `# Gatos
Informação geral sobre gatos.
## Comportamento
Gatos gostam de dormir o dia todo.
# Carros
Informação geral sobre carros.
## Combustível
Carros precisam de gasolina para funcionar.
`

// fakeNavigate escolhe o item cujo título+preview contém "gatos" ou "carros",
// conforme a pergunta — suficiente para exercitar a navegação em 2 estágios
// sem depender de Ollama.
func fakeNavigate(_ context.Context, prompt string) (string, error) {
	parts := strings.SplitN(strings.ToLower(prompt), "itens:", 2)
	if len(parts) != 2 {
		return "nenhum", nil
	}
	question := parts[0]
	itemsSection := parts[1]

	var keyword string
	switch {
	case strings.Contains(question, "gatos"):
		keyword = "gatos"
	case strings.Contains(question, "carros"):
		keyword = "carros"
	default:
		return "nenhum", nil
	}

	itemIndex := 0
	for _, line := range strings.Split(itemsSection, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || !strings.Contains(line, ".") {
			continue
		}
		itemIndex++
		if strings.Contains(line, keyword) {
			return strconv.Itoa(itemIndex), nil
		}
	}
	return "nenhum", nil
}

func TestTreeQueryNavigatesToLeaf(t *testing.T) {
	tr := newTreeWithNavigator(fakeNavigate)
	if err := tr.AddDocument(context.Background(), "doc1", "Documento", testDoc); err != nil {
		t.Fatalf("AddDocument: %v", err)
	}

	results, err := tr.Query(context.Background(), "o que os gatos fazem?", 3)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("esperava 1 nó folha, obtido %d: %+v", len(results), results)
	}
	if results[0].Title != "Comportamento" {
		t.Errorf("esperava chegar em 'Comportamento' (2 níveis de profundidade), obtido %q", results[0].Title)
	}
	if !strings.Contains(results[0].Content, "dormir") {
		t.Errorf("conteúdo do nó folha inesperado: %q", results[0].Content)
	}
}

func TestTreeQueryMaxDepthStopsEarly(t *testing.T) {
	tr := newTreeWithNavigator(fakeNavigate)
	if err := tr.AddDocument(context.Background(), "doc1", "Documento", testDoc); err != nil {
		t.Fatalf("AddDocument: %v", err)
	}

	// maxDepth=1: navega até o capítulo "Gatos" mas não desce até "Comportamento"
	// — devolve o próprio nó "Gatos" (com seu conteúdo próprio), não o leaf real.
	results, err := tr.Query(context.Background(), "o que os gatos fazem?", 1)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(results) != 1 || results[0].Title != "Gatos" {
		t.Fatalf("esperava parar em 'Gatos' com maxDepth=1, obtido: %+v", results)
	}
}

func TestKeywordFallbackStripsPunctuationFromQuestionWords(t *testing.T) {
	// Regressão: "motor?" (pontuação colada) não batia com "motor" no texto,
	// achado ao integrar tree no Router — o Ollama devolveu "1," (vírgula
	// sobrando) que caiu no fallback, e o fallback falhava por causa disso.
	failingNavigate := func(_ context.Context, _ string) (string, error) {
		return "", context.DeadlineExceeded
	}
	tr := newTreeWithNavigator(failingNavigate)
	if err := tr.AddDocument(context.Background(), "doc1", "Manual", testDoc); err != nil {
		t.Fatalf("AddDocument: %v", err)
	}

	results, err := tr.Query(context.Background(), "o que o manual diz sobre os gatos?", 3)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("esperava alcançar algum nó via fallback mesmo com pontuação colada em 'gatos?', obtido vazio")
	}
}

func TestParseChosenIndicesToleratesTrailingComma(t *testing.T) {
	children := []*Node{{Title: "A"}, {Title: "B"}}
	chosen, ok := parseChosenIndices("1,", children)
	if !ok {
		t.Fatal("esperava parseChosenIndices aceitar \"1,\" (vírgula sobrando), rejeitou")
	}
	if len(chosen) != 1 || chosen[0].Title != "A" {
		t.Errorf("esperava [A], obtido %+v", chosen)
	}
}

func TestKeywordFallbackUsedWhenNavigateFails(t *testing.T) {
	failingNavigate := func(_ context.Context, _ string) (string, error) {
		return "", context.DeadlineExceeded
	}
	tr := newTreeWithNavigator(failingNavigate)
	if err := tr.AddDocument(context.Background(), "doc1", "Documento", testDoc); err != nil {
		t.Fatalf("AddDocument: %v", err)
	}

	results, err := tr.Query(context.Background(), "gasolina carros", 3)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	found := false
	for _, r := range results {
		if r.Title == "Combustível" {
			found = true
		}
	}
	if !found {
		t.Errorf("esperava alcançar 'Combustível' via fallback de palavra-chave, resultados: %+v", results)
	}
}
