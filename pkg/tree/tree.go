// Package tree implementa o Vectorless RAG do uRag-go: um documento vira uma
// árvore hierárquica de headings markdown, e um LLM navega a árvore em vez de
// usar embeddings — bom para documentos estruturados (relatórios, manuais).
package tree

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"urag-go/internal/ollama"
	"urag-go/internal/openai"
)

// Config configura a Tree.
type Config struct {
	LLMProvider string // "ollama" ou "openai" (compatível: OpenAI e providers que implementem o mesmo formato)
	LLMModel    string
	// LLMBaseURL: override do endpoint. "" = default do provider (Ollama local,
	// ou api.openai.com para "openai") — obrigatório apontar pra providers
	// OpenAI-compatíveis que não são a OpenAI oficial (vLLM, LM Studio, etc).
	LLMBaseURL string
	// LLMAPIKey: usado só quando LLMProvider="openai".
	LLMAPIKey string
}

type navigateFunc func(ctx context.Context, prompt string) (string, error)

// Tree guarda uma árvore por documento (chave = ID do documento).
type Tree struct {
	docs     map[string]*Node
	navigate navigateFunc
}

// New cria uma Tree a partir de Config.
func New(cfg Config) (*Tree, error) {
	var navigate navigateFunc
	switch cfg.LLMProvider {
	case "ollama":
		navigate = func(ctx context.Context, prompt string) (string, error) {
			return ollama.Complete(ctx, cfg.LLMBaseURL, cfg.LLMModel, prompt, false)
		}
	case "openai":
		navigate = func(ctx context.Context, prompt string) (string, error) {
			return openai.Complete(ctx, cfg.LLMBaseURL, cfg.LLMAPIKey, cfg.LLMModel, prompt, false)
		}
	default:
		return nil, fmt.Errorf("tree: llm provider desconhecido: %q", cfg.LLMProvider)
	}
	return newTreeWithNavigator(navigate), nil
}

// NewWithNavigator cria uma Tree com uma função de navegação já pronta, sem
// resolver via Config.LLMProvider/LLMModel — útil para navegação customizada
// (provider fora de "ollama") ou para injetar um fake em testes.
func NewWithNavigator(navigate func(ctx context.Context, prompt string) (string, error)) *Tree {
	return newTreeWithNavigator(navigate)
}

// newTreeWithNavigator permite injetar um navigateFunc fake em testes, sem
// depender de Ollama rodando.
func newTreeWithNavigator(navigate navigateFunc) *Tree {
	return &Tree{docs: map[string]*Node{}, navigate: navigate}
}

// AddDocument parseia rawMarkdown em árvore e associa ao ID. ctx/error ficam
// reservados para quando a construção da árvore passar a envolver o LLM (ex:
// resumir seções muito longas) — hoje é parsing puro, nunca falha.
func (t *Tree) AddDocument(_ context.Context, id, title, rawMarkdown string) error {
	t.docs[id] = parseMarkdown(title, rawMarkdown)
	return nil
}

const navigatePrompt = `Você está navegando por um documento estruturado em árvore para responder a uma pergunta.
Escolha, dentre os itens abaixo, quais são relevantes para responder à pergunta.
Responda apenas com os números separados por vírgula (ex: "1,3") ou a palavra "nenhum" se nenhum for relevante.

Pergunta: %s

Itens:
%s`

const maxItemPreview = 80

// Query navega as árvores de todos os documentos em 2+ estágios (escolhe
// filhos relevantes recursivamente, até maxDepth ou até um leaf node) e
// devolve os nós folha alcançados. Não sintetiza resposta em texto — mesma
// filosofia de pkg/graph e pkg/router.
func (t *Tree) Query(ctx context.Context, question string, maxDepth int) ([]Node, error) {
	var results []Node
	for _, root := range t.docs {
		results = append(results, t.navigateNode(ctx, question, root, maxDepth)...)
	}
	return results, nil
}

func (t *Tree) navigateNode(ctx context.Context, question string, node *Node, depthLeft int) []Node {
	if len(node.Children) == 0 || depthLeft == 0 {
		return []Node{*node}
	}

	chosen := t.chooseChildren(ctx, question, node.Children)
	var results []Node
	for _, child := range chosen {
		results = append(results, t.navigateNode(ctx, question, child, depthLeft-1)...)
	}
	return results
}

func (t *Tree) chooseChildren(ctx context.Context, question string, children []*Node) []*Node {
	var items strings.Builder
	for i, c := range children {
		fmt.Fprintf(&items, "%d. %s: %s\n", i+1, c.Title, truncate(c.Content, maxItemPreview))
	}

	raw, err := t.navigate(ctx, fmt.Sprintf(navigatePrompt, question, items.String()))
	if err == nil {
		if chosen, ok := parseChosenIndices(raw, children); ok {
			return chosen
		}
	}

	// fallback: heurística de contagem de substring, não BM25 real.
	// ponytail: fallback é heurística de contagem de substring, trocar por BM25 de
	// verdade se a precisão do fallback virar problema real.
	return keywordFallback(question, children)
}

func parseChosenIndices(raw string, children []*Node) ([]*Node, bool) {
	raw = strings.ToLower(strings.TrimSpace(raw))
	if raw == "" {
		return nil, false
	}
	if raw == "nenhum" || raw == "none" {
		return nil, true
	}

	var chosen []*Node
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue // tolera vírgula sobrando no fim (ex: "1,"), comum em saída de LLM
		}
		n, err := strconv.Atoi(part)
		if err != nil || n < 1 || n > len(children) {
			return nil, false
		}
		chosen = append(chosen, children[n-1])
	}
	if len(chosen) == 0 {
		return nil, false
	}
	return chosen, true
}

var wordPunctuation = regexp.MustCompile(`[.,!?;:()"']+`)

func keywordFallback(question string, children []*Node) []*Node {
	var chosen []*Node
	for _, c := range children {
		haystack := strings.ToLower(c.Title + " " + c.Content)
		for _, word := range strings.Fields(strings.ToLower(question)) {
			word = wordPunctuation.ReplaceAllString(word, "")
			if len(word) < 3 {
				continue // ignora palavras curtas (artigos, preposições)
			}
			if strings.Contains(haystack, word) {
				chosen = append(chosen, c)
				break
			}
		}
	}
	return chosen
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
