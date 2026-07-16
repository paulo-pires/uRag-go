package tree

import "strings"

// Node é um nó da árvore hierárquica de um documento: um heading markdown
// (Title) com o texto sob ele até o próximo heading de nível igual/maior
// (Content), e os headings aninhados (Children).
type Node struct {
	Title    string
	Content  string
	Children []*Node
}

// parseMarkdown constrói a árvore a partir de headings markdown (#, ##, ###...),
// usando o algoritmo padrão de pilha: um heading de nível L vira filho do
// último heading aberto com nível < L. Níveis pulados (# seguido direto de ###)
// aninham normalmente sob o último heading de nível menor — sem heading
// sintético pro nível ausente.
func parseMarkdown(title, raw string) *Node {
	root := &Node{Title: title}
	stack := []*Node{root}
	levels := []int{0}

	var content strings.Builder
	flush := func() {
		text := strings.TrimSpace(content.String())
		content.Reset()
		if text == "" {
			return
		}
		top := stack[len(stack)-1]
		if top.Content != "" {
			top.Content += "\n"
		}
		top.Content += text
	}

	for _, line := range strings.Split(raw, "\n") {
		level, headingTitle, ok := parseHeadingLine(line)
		if !ok {
			content.WriteString(line)
			content.WriteString("\n")
			continue
		}

		flush()
		for len(stack) > 1 && levels[len(levels)-1] >= level {
			stack = stack[:len(stack)-1]
			levels = levels[:len(levels)-1]
		}

		node := &Node{Title: headingTitle}
		parent := stack[len(stack)-1]
		parent.Children = append(parent.Children, node)
		stack = append(stack, node)
		levels = append(levels, level)
	}
	flush()

	return root
}

// parseHeadingLine reconhece "# Título", "## Título", etc. (1 a 6 #'s seguidos
// de espaço). Qualquer outra linha não é heading.
func parseHeadingLine(line string) (level int, title string, ok bool) {
	trimmed := strings.TrimLeft(line, " \t")
	for level < len(trimmed) && trimmed[level] == '#' {
		level++
	}
	if level == 0 || level > 6 {
		return 0, "", false
	}
	rest := trimmed[level:]
	if !strings.HasPrefix(rest, " ") {
		return 0, "", false
	}
	return level, strings.TrimSpace(rest), true
}
