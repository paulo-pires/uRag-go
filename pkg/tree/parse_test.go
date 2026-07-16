package tree

import "testing"

func TestParseMarkdownNestsByHeadingLevel(t *testing.T) {
	raw := `Preâmbulo antes de qualquer heading.

# Capítulo 1
Texto do capítulo 1.

## Seção 1.1
Texto da seção 1.1.

## Seção 1.2
Texto da seção 1.2.

# Capítulo 2
Texto do capítulo 2.
`
	root := parseMarkdown("Documento", raw)

	if root.Title != "Documento" {
		t.Fatalf("esperava título raiz 'Documento', obtido %q", root.Title)
	}
	if root.Content != "Preâmbulo antes de qualquer heading." {
		t.Errorf("preâmbulo inesperado: %q", root.Content)
	}
	if len(root.Children) != 2 {
		t.Fatalf("esperava 2 capítulos, obtido %d", len(root.Children))
	}

	cap1 := root.Children[0]
	if cap1.Title != "Capítulo 1" || cap1.Content != "Texto do capítulo 1." {
		t.Errorf("capítulo 1 inesperado: %+v", cap1)
	}
	if len(cap1.Children) != 2 {
		t.Fatalf("esperava 2 seções em Capítulo 1, obtido %d", len(cap1.Children))
	}
	if cap1.Children[0].Title != "Seção 1.1" || cap1.Children[1].Title != "Seção 1.2" {
		t.Errorf("seções inesperadas: %+v", cap1.Children)
	}

	cap2 := root.Children[1]
	if cap2.Title != "Capítulo 2" || len(cap2.Children) != 0 {
		t.Errorf("capítulo 2 inesperado: %+v", cap2)
	}
}

func TestParseMarkdownSkippedHeadingLevel(t *testing.T) {
	raw := `# Nível 1
### Nível 3 (pulou o 2)
Conteúdo do nível 3.
## Nível 2 (irmão do nível 3, não filho)
Conteúdo do nível 2.
`
	root := parseMarkdown("Doc", raw)
	nivel1 := root.Children[0]
	if len(nivel1.Children) != 2 {
		t.Fatalf("esperava 2 filhos de Nível 1 (nível 3 e nível 2, ambos aninhados sob ele), obtido %d", len(nivel1.Children))
	}
	if nivel1.Children[0].Title != "Nível 3 (pulou o 2)" {
		t.Errorf("esperava Nível 3 como primeiro filho, obtido %q", nivel1.Children[0].Title)
	}
	if nivel1.Children[1].Title != "Nível 2 (irmão do nível 3, não filho)" {
		t.Errorf("esperava Nível 2 como segundo filho (irmão do nível 3), obtido %q", nivel1.Children[1].Title)
	}
	if len(nivel1.Children[0].Children) != 0 {
		t.Errorf("Nível 2 não deveria ter virado filho de Nível 3, mas Nível 3 tem filhos: %+v", nivel1.Children[0].Children)
	}
}

func TestParseHeadingLineRejectsNonHeadings(t *testing.T) {
	cases := []string{"", "texto normal", "#semespaço", "####### seteHashes"}
	for _, c := range cases {
		if _, _, ok := parseHeadingLine(c); ok {
			t.Errorf("parseHeadingLine(%q) deveria ser rejeitado, mas foi aceito", c)
		}
	}
}
