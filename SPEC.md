# uRag-go — Spec Técnica de Implementação

> Documento de trabalho para gerar código. Complementa o README (visão geral) com contratos concretos: interfaces Go, estrutura de pacotes e ordem de implementação.

## Status

- **Fase 1 (Vector RAG)**: completa. Ver seção "Fase 1" abaixo.
- **Fase 2, item 1 (filtros de metadata/documento)**: completo — `QueryFiltered` em `pkg/rag/rag.go`, `where`/`whereDocument` repassados ao `chromem-go`.
- **Fase 2, item 2 (Graph RAG)**: completo — `pkg/graph/` (extração via Ollama `/api/generate`, grafo em memória, BFS multi-hop). Validado ponta a ponta com `hf.co/unsloth/gemma-4-E2B-it-GGUF:IQ4_NL`.
- **Fase 2, item 3 (ANN/HNSW)**: completo — `Config.Index = "hnsw"` em `pkg/rag`, sobre um fork vendorizado de `coder/hnsw` em `internal/hnsw` (ver seção "Fase 2 — ANN (HNSW)" abaixo para o porquê do vendor).
- **Fase 2, item 4 (Router inteligente)**: completo — `pkg/router/` (classificação via LLM, fan-out em `AddDocuments`, despacho em `Query`). Default `Config.RouterModel = "granite4:micro-h"` (ver seção "Fase 2 — Router inteligente" para o porquê da escolha). Validado ponta a ponta com Ollama real nos dois branches (vector e graph).

**Fase 2 completa** (filtros, Graph RAG, ANN, Router).

- **Fase 3, item 1 (fusão vector+graph)**: completo — `Router.QueryFused` em `pkg/router/router.go`, roda as duas stores em paralelo (`sync.WaitGroup`) e devolve `FusedResult{Vector, Graph}` sem tentar unificar num score só (não existe métrica comparável entre similaridade de embedding e aresta de grafo). `Query` (dispatch único) continua o caminho padrão, inalterado. Validado ponta a ponta com Ollama real.
- **Fase 3, item 2 (roteamento multi-estratégia)**: código completo (`StrategyBoth`, testado), mas `both` não é confiável no modelo default `granite4:micro-h` — ver seção 4 da subseção correspondente.
- **Fase 3, item 3 (few-shot no prompt)**: completo, com uma regressão real encontrada e corrigida durante a validação — ver seção 4 da subseção correspondente.
- **Fase 3, item 4 (Vectorless RAG)**: completo — `pkg/tree/` (parser markdown→árvore, navegação em 2+ estágios via LLM, fallback de palavra-chave). Validado ponta a ponta com Ollama real, incluindo um caso de correspondência semântica não-literal.
- **Fase 3, item 5 (Text-to-SQL)**: completo — `pkg/sql/` (SQLite via `modernc.org/sqlite`, introspecção de schema, validação de segurança obrigatória pré-execução). Validado ponta a ponta: 1 pergunta correta, 1 com erro real de correspondência de valor (documentado — limitação conhecida, não bug).

**Todos os itens da ordem combinada pelo usuário estão completos.**

- **Fase 4 (integrar Tree e SQL no Router)**: completo — `Strategy` ganhou `StrategyTree`/`StrategySQL`, `Config`/`QueryResult` estendidos, `Router.AddMarkdownDocument` novo (Tree não usa `[]rag.Document`). Ver seção "Fase 4 — Router com 4 stores" abaixo para detalhes, inclusive um bug real de navegação encontrado e corrigido durante a validação.

- **Fase 5 (CLI para graph/tree/sql/router)**: completo — `cmd/urag/main.go` ganhou `graph ask`, `tree ask`, `sql query`, `router ask`. `add`/`query` (Vector RAG) inalterados. Graph e Tree são in-memory (decisão já registrada), então os comandos novos são "single-shot": carregam e consultam na mesma invocação, diferente do padrão `add` depois `query` em processos separados que só faz sentido pra Vector RAG (que persiste em disco). Validado: build/vet/test limpos + smoke test de dispatch/usage sem LLM real. **Não rodei um novo e2e com Ollama real pra cada subcomando novo** — são wrappers finos sobre chamadas já exaustivamente validadas nesta sessão (mesmo `graph.New`/`AddDocuments`/`Query` etc.), e o custo da sessão já está muito alto; risco residual é baixo (é só parsing de flags), mas fica registrado como não 100% coberto por teste manual real.

Próximos passos possíveis ficam nas seções "Fora de escopo"/"Decisões" de cada subseção (ex: v2 do Text-to-SQL com amostra de valores de coluna, modelo maior para classificação confiável de `both`/`tree`/`sql` em 5 categorias).

- **Fase 6 (Interface MCP)**: completa — `pkg/mcpserver/` (7 tools: vector/graph/tree add+query, sql_query condicional), `github.com/modelcontextprotocol/go-sdk` como SDK oficial, `urag mcp serve` no CLI. Ver seção "Fase 6" abaixo para detalhes, inclusive o `Close()` adicionado a `sql.Store` (gap descoberto durante os testes) e a validação e2e real feita com um cliente MCP via `CommandTransport` (sem precisar de Ollama, já que `vector_add`/`vector_query` não dependem de LLM).

---

# Fase 1: MVP Vector RAG

---

## 1. Escopo do MVP

Dentro:
- Wrapper fino sobre `chromem-go` (não reimplementar vector store).
- Providers de embedding: Ollama (local) + OpenAI (hosted) — os dois mais usados, resto via interface aberta.
- API de coleções/documentos (Add, Query, Delete).
- Persistência via o que `chromem-go` já oferece (gob).
- CLI mínima: `urag add`, `urag query`.

Fora (fica para Fase 2+, não escrever stub agora):
- Graph RAG, Vectorless RAG, Text-to-SQL, Router, módulos RoleRAG, MCP server, API HTTP.

**Por quê**: essas quatro áreas são projetos por si só; código morto/interface especulativa antes de o vector path funcionar é custo sem retorno.

---

## 2. Estrutura de diretórios (MVP)

```
uRag-go/
├── cmd/
│   └── urag/            # CLI única (add/query)
│       └── main.go
├── pkg/
│   └── rag/              # API pública
│       ├── rag.go        # struct UnifiedRAG, Config, New()
│       ├── document.go   # tipo Document
│       └── vector.go     # wrapper chromem-go
├── internal/
│   └── embedding/
│       ├── ollama.go
│       └── openai.go
├── go.mod
└── README.md
```

Pacotes `graph/`, `tree/`, `sql/`, `router/`, `pipeline/`, `mcp/` só nascem quando a Fase correspondente começar — evita interfaces "para depois" sem implementação real.

---

## 3. Contratos Go (núcleo)

```go
// pkg/rag/document.go
package rag

type Document struct {
    ID      string
    Content string
    Source  string
    Meta    map[string]string
}

type SearchResult struct {
    Document Document
    Score    float32
}
```

```go
// pkg/rag/rag.go
package rag

import "context"

type Config struct {
    EmbeddingProvider string // "ollama" | "openai"
    EmbeddingModel    string
    PersistPath       string // "" = in-memory
}

type UnifiedRAG struct {
    // wraps *chromem.DB internamente — não expor chromem no contrato público
}

func New(cfg Config) (*UnifiedRAG, error)
func (u *UnifiedRAG) AddDocuments(ctx context.Context, docs []Document) error
func (u *UnifiedRAG) Query(ctx context.Context, question string, topK int) ([]SearchResult, error)
```

Ponto de decisão: `UnifiedRAG` hoje só embrulha o Vector RAG. Quando Graph/Tree/SQL entrarem, o `Query` ganha um router — não desenhar essa interface agora (YAGNI), só deixar o nome do método estável.

```go
// internal/embedding/*.go
package embedding

import "context"

// Assinatura compatível com chromem.EmbeddingFunc — não reinventar.
type Func func(ctx context.Context, text string) ([]float32, error)

func Ollama(model string) Func
func OpenAI(apiKey, model string) Func
```

---

## 4. Dependências

```go
module github.com/<org>/uRag-go

go 1.22

require (
    github.com/philippgille/chromem-go v0.7.0
)
```

CLI usa apenas `flag` da stdlib no MVP — sem Cobra/urfave até a superfície de comandos crescer além de `add`/`query`.

---

## 5. CLI mínima

```
urag add    -source <path> -db <path>
urag query  -q "pergunta" -db <path> -k 5
```

Sem servidor HTTP nesta fase — Gin/Echo entram junto com o Router (Fase 2/API Gateway do README).

---

## 6. Ordem de implementação

1. `go.mod` + `pkg/rag/document.go`
2. `internal/embedding/ollama.go` (provider mais barato para testar local)
3. `pkg/rag/vector.go` — wrapper chromem-go (New, AddDocuments, Query)
4. `pkg/rag/rag.go` — monta Config → UnifiedRAG
5. `cmd/urag/main.go` — CLI chamando o pacote `rag`
6. Teste manual: `urag add` com 2-3 docs + `urag query` contra Ollama local
7. `internal/embedding/openai.go` (segundo provider, valida que a interface `Func` generaliza)

Critério de "Fase 1 pronta": CLI funcional ponta a ponta com Ollama, sem Graph/Tree/SQL no caminho.

---

## 7. Testes

Cada arquivo não-trivial leva um `_test.go` mínimo (sem framework, `testing` da stdlib):
- `vector_test.go`: Add + Query retorna o doc esperado (usa embedding fake determinístico, não chama Ollama de verdade).
- `embedding/ollama_test.go`: skip se `OLLAMA_HOST` não setado (evita quebrar CI sem Ollama rodando).

---

## 8. Não fazer agora

- Não criar `router/`, `graph/`, `tree/`, `sql/`, `mcp/` vazios "para manter a estrutura do README" — cria quando a Fase 2 começar.
- Não adicionar Gin/Echo antes de existir um segundo consumidor além da CLI.
- Não implementar múltiplos formatos de persistência — `chromem-go` já resolve isso.

---

# Fase 2 — Graph RAG (MVP)

## 1. Gap novo: extração precisa de LLM, não só embedding

Vector RAG (Fase 1) só usa `EmbeddingFunc` (texto → vetor). Graph RAG precisa de um LLM que **raciocina**: lê um documento e devolve entidades + relações em JSON. `chromem-go` não oferece isso — é uma capacidade nova.

Decisão: função de completion mínima, sem SDK novo, batendo direto no endpoint HTTP do Ollama (`/api/generate`, `format: "json"`), do mesmo jeito que `chromem-go` já bate no endpoint de embeddings do Ollama internamente. Sem dependência nova.

```go
// pkg/graph/extract.go
package graph

import "context"

// completionFunc pede uma resposta em JSON para um prompt. Implementação real
// chama Ollama /api/generate; testes injetam uma fake, mesmo padrão do fakeEmbedding
// em pkg/rag/vector_test.go.
type completionFunc func(ctx context.Context, prompt string) (string, error)
```

## 2. Escopo dentro do MVP

- Extração de entidades + relações via 1 chamada LLM por `Document` (sem chunking — documento grande é problema de uma fase de pipeline futura, não desta).
- Grafo em memória: `map[string]Entity` (chave = nome normalizado) + slice de `Relation`. Sem dependência de grafo (gonum etc.) — é só um índice de adjacência, não precisamos de algoritmos de grafo genéricos.
- Query: (1) acha entidades "seed" cujo nome aparece na pergunta (substring, case-insensitive — sem embedding de entidades no MVP); (2) expande por BFS até `hops` relações de distância; (3) devolve as `Relation`s tocadas como contexto — **não sintetiza resposta em texto** (isso é o "Answer Reasoner" do pipeline RoleRAG, fora de escopo aqui).
- Dedup de entidade: só por nome normalizado (lowercase + trim). Sem fuzzy match / merge semântico.

## 3. Fora do MVP (explícito)

- Persistência em disco do grafo (fica in-memory; Vector RAG já tem persistência, Graph RAG ainda não precisa).
- Integração no `UnifiedRAG.Query` / Router (só faz sentido quando o Router existir — próximo item da ordem combinada).
- MCP server.
- Scores/confiança em relações.
- Entity resolution via embedding.

## 4. Contratos Go

```go
// pkg/graph/graph.go
package graph

import "context"

type Entity struct {
    Name string
    Type string // livre, o que o LLM extrair (ex: "Pessoa", "Empresa")
}

type Relation struct {
    Source   string // nome normalizado da entidade origem
    Target   string
    Relation string // texto livre extraído pelo LLM (ex: "trabalha_em")
    DocID    string // proveniência
}

type Config struct {
    LLMProvider string // "ollama" (único suportado no MVP)
    LLMModel    string
}

type GraphStore struct {
    // entities map[string]Entity, relations []Relation, adjacency map[string][]int — internos
}

func New(cfg Config) (*GraphStore, error)
func (g *GraphStore) AddDocuments(ctx context.Context, docs []rag.Document) error
func (g *GraphStore) Query(ctx context.Context, question string, hops int) ([]Relation, error)
```

`AddDocuments` reaproveita `rag.Document` (já existe em `pkg/rag`) — não criar um tipo `Document` paralelo.

## 5. Prompt de extração (v1, fixo)

Um único prompt fixo (sem few-shot, sem configuração), pedindo JSON estrito:

```
Extraia entidades e relações do texto abaixo. Responda apenas em JSON:
{"entities":[{"name":"...","type":"..."}],"relations":[{"source":"...","target":"...","relation":"..."}]}

Texto: <conteúdo do documento>
```

Se o LLM devolver JSON inválido: descarta o documento daquela extração e loga erro — não trava o `AddDocuments` inteiro (mesma filosofia de "não travar em degradação parcial" que já vale pro resto do projeto).

## 6. Ordem de implementação

1. `pkg/graph/extract.go` — `completionFunc`, implementação real via HTTP para Ollama `/api/generate`.
2. `pkg/graph/graph.go` — `Entity`, `Relation`, `Config`, `GraphStore`, `New`.
3. `AddDocuments`: extrai + funde no grafo em memória (dedup por nome normalizado).
4. `Query`: seed match por substring + BFS até `hops`.
5. Teste com `completionFunc` fake determinística (sem Ollama real) para `AddDocuments`/BFS.
6. Teste manual ponta a ponta com Ollama real: 2-3 documentos com entidades relacionadas, pergunta multi-hop.

Critério de "Graph RAG MVP pronto": `graph.New` + `AddDocuments` + `Query` funcionando ponta a ponta com Ollama real, devolvendo relações corretas para uma pergunta 2-hops.

## 7. Modelo de extração

Decidido pelo usuário: `hf.co/unsloth/gemma-4-E2B-it-GGUF:IQ4_NL`. **Não está pulado localmente ainda** — rodar `ollama pull hf.co/unsloth/gemma-4-E2B-it-GGUF:IQ4_NL` antes do teste ponta a ponta da seção 6.6. Configurável via `Config.LLMModel`, sem hardcode fixo no código — esse é só o default usado nos exemplos/testes manuais.

---

# Fase 2 — ANN (HNSW)

## 1. Decisão: lib pronta, não hand-rolled

HNSW correto (níveis, poda de vizinhos, busca em camada) é não-trivial — bugs de recall/precisão são fáceis de introduzir e difíceis de notar em testes pequenos. Decisão: usar uma lib pronta em vez de implementar o algoritmo.

## 2. Trilha de libs testadas (para não repetir a investigação)

| Lib | Resultado |
|---|---|
| `github.com/coder/hnsw` | API limpa (`Graph[K]`, `Node[K]` com chave genérica), mas **não compila no Windows**: `encode.go` chama `renameio.TempFile` incondicionalmente, e a v1 de `google/renameio` não implementa `TempFile` para Windows (`// +build !windows` sem equivalente). Confirmado lendo o código-fonte: não há build tag em `coder/hnsw` nem em `renameio` que contorne isso — `-tags=hnsw` testado e não muda nada (nenhum arquivo em nenhuma das duas libs usa essa tag). |
| `github.com/TFMV/hnsw` | Fork do `coder/hnsw`, herdou o mesmo bug de `renameio.TempFile`. |
| `github.com/fogfish/hnsw` | Compila, mas API mais baixo nível — exige embutir a chave (doc ID) dentro do próprio tipo `Vector` e implementar uma interface `Surface` de distância. Lib imatura (v0.0.5). |
| `github.com/habedi/hann` | Compila limpo no Windows, mas a distance function usa **CGO** (`import "C"`, flags `-mavx`, chama `simd_cosine_distance` em C) — quebra o objetivo de binário único sem dependências externas e exige gcc/MinGW em qualquer máquina que for compilar. |
| `github.com/DotNetAge/govector` | Pure Go, CGO-free, mas exige Go ≥ 1.25.1 (ambiente tinha 1.24.2 instalado). |

## 3. Solução: fork vendorizado de `coder/hnsw`

`internal/hnsw/` — cópia de `coder/hnsw` v0.6.1 (CC0, ver `internal/hnsw/LICENSE`), com uma única mudança: `SavedGraph`/`LoadSavedGraph`/`(*SavedGraph).Save` removidos de `encode.go` (era a única coisa que chamava `renameio.TempFile`). `Export`/`Import` (que não dependem de `renameio`) continuam intactos. Detalhes em `internal/hnsw/NOTICE.md`.

Justificativa de escopo: uRag-go já não persiste o índice ANN (`Index=hnsw` é incompatível com `PersistPath`, ver seção 5), então a função removida nunca seria chamada mesmo se existisse.

## 4. Arquitetura

O índice ANN é uma camada opcional **interna** ao `vectorStore` (`pkg/rag/vector.go`) — invisível para `rag.go` e para o CLI. Liga/desliga via `Config.Index`:

```go
type Config struct {
    ...
    Index string // "" (exaustivo, default, via chromem-go) | "hnsw" (aproximado)
}
```

- `add()`: quando `Index=hnsw`, calcula o embedding manualmente (via o mesmo `embeddingFunc` do `chromem-go`) para popular tanto `chromem.Document.Embedding` (evita recomputar) quanto o grafo `hnsw.Graph[string]`.
- `query()`: roteia internamente para `queryExhaustive` (comportamento atual, via `chromem-go`) ou `queryANN` (embeda a pergunta, busca no grafo, busca o documento completo via `collection.GetByID`).
- `where` (metadata) funciona nos dois caminhos — no caminho ANN é aplicado como filtro pós-busca (`matchesWhere`), já que o índice não entende metadata.
- `whereDocument` (`$contains` etc.) **não é suportado** no caminho ANN — retorna erro explícito em vez de ignorar silenciosamente (evita comportamento surpreendente).

## 5. Fora do MVP (explícito)

- Persistência do índice ANN: `Index=hnsw` + `PersistPath` retorna erro na criação (`New`/`newVectorStoreWithEmbedding`). Sem isso, um restart perderia o índice ANN silenciosamente enquanto o `chromem-go` recarregaria os documentos normalmente — um bug de degradação silenciosa que preferimos bloquear explicitamente a deixar acontecer.
- IVFFlat: não implementado — HNSW já cobre o caso de uso (aproximação para escala), IVFFlat ficaria redundante sem um motivo concreto para escolher um sobre o outro.

Critério de "ANN pronto": `rag.New(Config{Index: "hnsw", ...})` + `AddDocuments` + `Query`/`QueryFiltered` funcionando ponta a ponta com Ollama real, resultado top-1 correto e filtro `where` funcionando.

---

# Fase 2 — Router inteligente

## 1. Por que só agora

Router escolhe entre Vector RAG e Graph RAG — só faz sentido com os dois existindo e populáveis juntos. Ambos existem agora (`pkg/rag`, `pkg/graph`).

## 2. Restrição de arquitetura: não dá pra colocar o Router dentro de `pkg/rag`

`pkg/graph` já importa `pkg/rag` (usa `rag.Document`). Se `pkg/rag` importasse `pkg/graph` pra montar o router ali dentro, seria import cycle (`rag → graph → rag`). Solução: pacote novo, `pkg/router`, que importa os dois e orquestra por cima — não mexe em `pkg/rag` nem `pkg/graph`.

## 3. Escopo dentro do MVP

- `Router.AddDocuments` faz fan-out: chama `rag.UnifiedRAG.AddDocuments` e `graph.GraphStore.AddDocuments` com os mesmos documentos. Sem escolha seletiva de que doc vai pra qual store — YAGNI até aparecer um caso real que precise disso.
- `Router.Query` faz 1 chamada LLM de classificação (prompt fixo, pede "vector" ou "graph"), despacha pra store escolhida, devolve o resultado **nativo** daquela store — sem sintetizar resposta final em texto (mesma filosofia do Graph RAG: isso é o "Answer Reasoner" do pipeline RoleRAG, fora de escopo aqui).
- Fallback se a classificação vier ambígua/vazia: default pra `"vector"` (mais barato; exhaustive/HNSW já é rápido — degradação segura em vez de erro, mesma filosofia de "não travar" já usada no resto do projeto).

## 4. Refactor pequeno antes: HTTP de completion do Ollama vira compartilhado

Hoje `pkg/graph/extract.go` tem uma função não-exportada que bate em `/api/generate`. O Router precisa da mesma coisa pra classificar. Extrair para `internal/ollama/complete.go` (função genérica), usada tanto por `pkg/graph` (troca a implementação interna, mesmo comportamento, testes continuam passando) quanto por `pkg/router`. Evita duplicar ~40 linhas de boilerplate HTTP.

## 5. Contratos Go

```go
// internal/ollama/complete.go
package ollama

func Complete(ctx context.Context, baseURL, model, prompt string) (string, error)
```

```go
// pkg/router/router.go
package router

import (
    "context"

    "urag-go/pkg/graph"
    "urag-go/pkg/rag"
)

type Strategy string

const (
    StrategyVector Strategy = "vector"
    StrategyGraph  Strategy = "graph"
)

type Config struct {
    Vector      rag.Config
    Graph       graph.Config
    RouterModel string // modelo Ollama para classificação
}

// QueryResult carrega o resultado nativo da store escolhida — só um dos dois
// campos vem preenchido, conforme Strategy.
type QueryResult struct {
    Strategy Strategy
    Vector   []rag.SearchResult
    Graph    []graph.Relation
}

type Router struct {
    vector *rag.UnifiedRAG
    graph  *graph.GraphStore
}

func New(cfg Config) (*Router, error)
func (r *Router) AddDocuments(ctx context.Context, docs []rag.Document) error
func (r *Router) Query(ctx context.Context, question string, topK int) (QueryResult, error)
```

## 6. Prompt de classificação (fixo, v1)

```
Classifique a pergunta abaixo em "vector" ou "graph":
- "vector": busca por similaridade de conteúdo/texto, factual, um único documento resolve.
- "graph": pergunta sobre relações entre entidades, multi-hop (ex: "quem trabalha em X e onde X fica").
Responda apenas com a palavra: vector ou graph.

Pergunta: <question>
```

## 7. Fora do escopo

- Fusão/reranking de resultados vector+graph numa única resposta.
- Roteamento por múltiplas estratégias simultâneas (hoje é 1 pergunta → 1 estratégia).
- Few-shot ou fine-tuning no prompt de classificação — 1 chamada simples, sem exemplos.
- `AddDocuments` seletivo por store.
- Vectorless RAG / Text-to-SQL como opções de roteamento (não existem ainda).

## 8. Ordem de implementação

1. Extrair `internal/ollama/complete.go`; atualizar `pkg/graph/extract.go` para usá-lo (testes existentes de `pkg/graph` continuam passando sem alteração).
2. `pkg/router/router.go` — tipos + `New`.
3. `AddDocuments` (fan-out vector + graph).
4. `Query` — classifica, despacha, devolve `QueryResult`.
5. Teste com completion fake (classificação determinística por conteúdo do prompt, sem Ollama real) cobrindo os dois branches + fallback ambíguo.
6. Teste manual ponta a ponta com Ollama real: 1 pergunta que deveria cair em vector, 1 que deveria cair em graph.

Critério de "Router pronto": `router.New` + `AddDocuments` + `Query` funcionando ponta a ponta, classificando corretamente as duas perguntas de teste do passo 6.

## 9. Modelo de classificação

Testados 3 modelos leves já instalados contra as duas perguntas de teste do passo 6:

| Modelo | Resultado |
|---|---|
| `qwen3.5:0.8b` | **Bug de parsing, não de classificação**: é um modelo "thinking" — a resposta final vai pro campo `thinking` da API do Ollama, não pro campo `response` que `internal/ollama.Complete` lê. `response` sempre veio vazio, então tudo caía no fallback `vector`. |
| `qwen2.5-coder:3b` | Sem problema de parsing, mas errou a classificação multi-hop (respondeu "vector" pra pergunta que deveria ser "graph") — é um modelo de código, não generalista o bastante para essa tarefa. |
| `granite4:micro-h` | Acertou os dois casos de teste, sem problema de parsing. **Escolhido como `defaultRouterModel`.** |

Configurável via `Config.RouterModel` — o default é só o que os testes manuais validaram, não uma imposição.

---

# Fase 3 — Roteamento multi-estratégia

## 1. Escopo

Hoje `classifyStrategy` só reconhece `"vector"`/`"graph"`; qualquer outra coisa (ambíguo, erro) cai em `StrategyVector` por segurança. Multi-estratégia adiciona uma terceira saída explícita — `StrategyBoth` — para perguntas que genuinamente precisam das duas (ex: "quem trabalha na Ignus e o que a empresa faz?": relação + busca factual).

- `Strategy` ganha `StrategyBoth = "both"`.
- `classifyPrompt` passa a oferecer 3 opções (`vector`/`graph`/`both`), com 1 exemplo do caso `both`.
- `classifyStrategy` reconhece as 3; qualquer resposta que não bater em nenhuma continua caindo em `StrategyVector` (degradação ambígua = mais barato, não = "tenta as duas" — só usa `both` quando o LLM pede explicitamente).
- `Query` ganha um terceiro branch: quando `StrategyBoth`, chama a mesma lógica interna de `QueryFused` e devolve `QueryResult{Strategy: StrategyBoth, Vector: ..., Graph: ...}` (reaproveita o código, não duplica).

## 2. Fora de escopo

- Threshold de confiança / múltiplas chamadas de classificação — continua sendo 1 chamada LLM, resposta determinística.
- N estratégias além de vector/graph (Vectorless/SQL não entram no roteamento até existirem).

## 3. Ordem de implementação

1. Adicionar `StrategyBoth`, atualizar `classifyPrompt` com o terceiro caso + exemplo.
2. `classifyStrategy`: reconhecer `"both"`.
3. `Query`: extrair a lógica de `QueryFused` para um método interno compartilhado (`queryBoth`), usado tanto por `QueryFused` (sempre) quanto por `Query` (quando classificado como `both`).
4. Teste com classify fake retornando `"both"` — confirma que `Query` (não só `QueryFused`) devolve os dois preenchidos.
5. Teste manual: uma pergunta desenhada para ser ambígua o bastante pra classificar como `both` de propósito.

Critério de pronto: `Query` classificando `both` corretamente numa pergunta real via Ollama, sem regressão nos branches `vector`/`graph` existentes.

## 4. Status: código completo, `both` não é confiável no modelo default

`StrategyBoth`, `classifyStrategy` e `Query`/`queryBoth` implementados e testados (`TestRouterQueryDispatchesToBoth` prova que o roteamento funciona corretamente quando a classificação vem `"both"`). **Mas** `granite4:micro-h` (3B) não classifica `"both"` de forma confiável em perguntas compostas ambíguas — testado com múltiplas variações da mesma pergunta (`"quem trabalha na Ignus e o que e python?"`), sempre voltou `"graph"`. Diagnóstico direto via curl confirmou: o modelo tem viés forte pra `"graph"` assim que reconhece o padrão "quem trabalha em X", mesmo quando a segunda metade da pergunta é claramente factual. É limite de capacidade do modelo pequeno pra discriminação em 3 classes, não bug de código. Fica registrado como limitação conhecida — trocar por um modelo maior se a classificação `both` precisar funcionar de verdade em produção.

---

# Fase 3 — Few-shot no prompt de classificação

## 1. Escopo

Só editar `classifyPrompt`: adicionar 3-4 exemplos fixos (pergunta → resposta esperada) cobrindo os 3 casos (`vector`, `graph`, e `both` se o item anterior já tiver entrado). Sem mecanismo novo — não é RAG sobre exemplos, não é seleção dinâmica de few-shot por similaridade (isso seria over-engineering pra uma classificação de 3 categorias). É só texto a mais no prompt fixo já existente.

Exemplos usados devem ser **diferentes** dos fixtures de teste (`Maria`/`Ignus`) pra não mascarar overfitting — usar domínio genérico (ex: filme, livro, empresa fictícia diferente).

## 2. Fora de escopo

- Few-shot dinâmico/selecionado por embedding.
- Fine-tuning do modelo.
- Configuração de quantos exemplos usar — fixo no código.

## 3. Ordem de implementação

1. Escrever os exemplos, adicionar ao `classifyPrompt`.
2. Rodar de novo os testes manuais das fases anteriores (Router + multi-estratégia) pra confirmar que não regrediu nenhum caso que já funcionava.
3. Se der pra achar 1-2 perguntas que o modelo errava antes e acerta agora, documentar no SPEC como evidência de que valeu a pena (senão, é só uma mudança de prompt sem verificação real de ganho).

Critério de pronto: mesmos testes manuais das fases 1 (roteamento) e 2 (multi-estratégia) passando, sem regressão.

## 4. Status: regressão real encontrada e corrigida

Primeira versão dos exemplos few-shot **removeu** o exemplo inline que já existia na descrição da categoria "graph" (substituindo por só o bloco de exemplos separado). Isso quebrou a classificação `graph` que já funcionava — a pergunta "em que pais fica a empresa onde Maria trabalha?" (que classificava certo antes) passou a cair em `vector`. Diagnosticado via curl direto comparando as duas versões do prompt lado a lado. Correção: manter o exemplo inline na descrição da categoria **e** o bloco de few-shot — funcionam juntos, não são substitutos um do outro. Reconfirmado ponta a ponta: `vector` e `graph` voltaram a classificar certo. `both` continua não confiável (ver seção 4 da fase anterior — limitação do modelo, não do prompt).

---

# Fase 3 — Vectorless RAG

## 1. Conceito (do README original)

Documento vira uma árvore hierárquica (capítulos → seções); o LLM navega a árvore em 2 estágios (escolhe capítulos relevantes, depois desce nas subseções) em vez de usar embeddings. Bom pra documentos estruturados (relatórios, manuais, leis) onde chunking por embedding perde contexto de estrutura.

## 2. Gap novo: de onde vem a árvore

Hoje `rag.Document`/`graph`'s `AddDocuments` recebem uma lista de documentos já "achatados" (um `Content` string cada). Vectorless RAG precisa de **estrutura hierárquica**, que não existe nesse modelo. Decisão: `pkg/tree` não reaproveita `AddDocuments([]rag.Document)` — recebe o documento bruto (markdown) e constrói a árvore internamente, parseando headings (`#`, `##`, `###`).

```go
// pkg/tree/tree.go
package tree

type Node struct {
    Title    string
    Content  string  // texto sob esse heading, antes do próximo heading de nível igual/maior
    Children []*Node
}

type Config struct {
    LLMProvider string // "ollama"
    LLMModel    string
}

type Tree struct {
    // docs map[string]*Node — raiz por doc ID; completion func — internos
}

func New(cfg Config) (*Tree, error)
func (t *Tree) AddDocument(ctx context.Context, id string, rawMarkdown string) error
func (t *Tree) Query(ctx context.Context, question string, maxDepth int) ([]Node, error)
```

## 3. Navegação (2 estágios, conforme README)

- Estágio 1: LLM recebe os títulos (+ resumo curto, ex: primeiras N palavras) dos nós de nível 1 de todos os documentos, escolhe quais são relevantes.
- Estágio 2: dentro de cada nó escolhido, repete o mesmo processo com os filhos, até `maxDepth` ou até não ter mais filhos (leaf node).
- Retorna os nós folha alcançados como contexto — sem sintetizar resposta (mesma filosofia de Graph RAG/Router).
- **Fallback simplificado, não BM25 de verdade**: se o LLM não devolver uma escolha parseável em algum estágio, cai para contagem simples de palavras-chave da pergunta presentes no título/conteúdo do nó (substring count) — não é TF-IDF/BM25 real. `ponytail: fallback é heurística de contagem de substring, trocar por BM25 de verdade se a precisão do fallback virar problema real.`

## 4. Fora de escopo

- Parsers para outros formatos (PDF, DOCX) — só markdown com headings `#`/`##`/`###`.
- Múltiplos documentos fundidos numa árvore só — 1 documento = 1 árvore (identificada pelo ID).
- BM25 real — fallback é heurística simples (ver acima).
- Persistência da árvore — in-memory, mesma decisão já tomada pro índice ANN.
- Integração no Router (`pkg/router`) — cria um 3º branch de `Strategy` só depois que isso existir e fizer sentido rotear pra ele.

## 5. Ordem de implementação

1. Parser de markdown → `Node` tree (sem LLM ainda) — função pura, testável sem Ollama.
2. `pkg/tree/tree.go` — `Config`, `Tree`, `New`, `AddDocument`.
3. Prompt de navegação estágio 1 + estágio 2 (mesmo padrão de `completionFunc` já usado em `pkg/graph`).
4. `Query` — navega, aplica fallback de heurística quando o LLM não responde algo parseável.
5. Teste do parser markdown→tree com casos fixos (sem LLM).
6. Teste de `Query` com completion fake determinística (navegação previsível).
7. Teste manual ponta a ponta: documento markdown com 2-3 níveis de heading, pergunta que só uma seção específica resolve.

Critério de pronto: `tree.New` + `AddDocument` + `Query` funcionando ponta a ponta com Ollama real, alcançando o nó folha certo pra uma pergunta de teste.

## 6. Status: completo, validado ponta a ponta

Implementado em `pkg/tree/` (`parse.go` + `tree.go`), com `granite4:micro-h` como modelo de navegação — mesma escolha já validada no Router, funcionou bem aqui também. Teste manual real: documento com 2 níveis (capítulo → seção), 2 perguntas ("o que os gatos comem?", "que combustível os carros usam?") — a navegação alcançou as seções certas nos dois casos, inclusive um caso de correspondência semântica não-literal ("comem" → seção "Alimentação", sem overlap de palavra exata). 9 testes automatizados (3 parser puro + 3 `Query` com navigateFunc fake + 3 embutidos cobrindo maxDepth e fallback de palavra-chave).

---

# Fase 3 — Text-to-SQL

## 1. Conceito (do README original)

Converte pergunta em linguagem natural pra SQL executável contra um banco relacional, executa, devolve resultado. É o item com **maior risco operacional** dos 4 — executa SQL gerado por LLM contra um banco real.

## 2. Gap novo: primeira dependência de driver de banco

Nenhum pacote atual conecta a bancos externos. Precisa de `database/sql` (stdlib) + 1 driver. **Decisão em aberto, é sua**: qual banco suportar primeiro.

| Opção | Driver | CGO? |
|---|---|---|
| PostgreSQL | `github.com/jackc/pgx` (ou `lib/pq`) | Não — cliente TCP puro Go |
| SQLite | `modernc.org/sqlite` | Não (é a versão pure-Go/transpilada; `mattn/go-sqlite3` usaria CGO — **não usar**, mesmo problema que já tivemos com `coder/hnsw`) |
| MySQL | `github.com/go-sql-driver/mysql` | Não |

Dado o histórico recente (CGO quebrou a escolha de lib do ANN), qualquer uma dessas três é segura — nenhuma usa CGO. A decisão real é qual banco faz sentido pro seu caso de uso.

## 3. Escopo dentro do MVP

- 1 banco suportado no v1 (o escolhido acima) — sem abstração multi-driver até um segundo banco ser um caso real (YAGNI, mesma lógica já aplicada em outras partes do projeto).
- `New(cfg)`: conecta, faz introspecção do schema (`information_schema` ou equivalente do banco escolhido) — tabelas + colunas, usado como contexto no prompt.
- `Query(ctx, question) (rows []map[string]any, sql string, error)`: 1 chamada LLM gera SQL a partir do schema + pergunta, valida, executa, devolve linhas **e** o SQL gerado (transparência — você vê o que rodou).
- **Validação de segurança obrigatória, não opcional**: o SQL gerado é entrada não-confiável cruzando pra um banco real.
  - Rejeitar qualquer statement que não comece com `SELECT` (sem `INSERT`/`UPDATE`/`DELETE`/`DROP`/etc.).
  - Rejeitar múltiplos statements (`;` no meio da string, fora de string literal) — evita stacked-query injection.
  - Se o driver/banco escolhido suportar, rodar em transação read-only ou com role de banco somente-leitura.

## 4. Fora de escopo

- Cache de resultados frequentes / RAG sobre templates de query passados (a versão "chique" do README, com Qdrant + re-ranking por frequência/taxa de sucesso) — é um subsistema de retrieval próprio, fora de escopo do v1.
- Múltiplos bancos simultâneos.
- Escrita (INSERT/UPDATE/DELETE) — só leitura.
- Joins/queries complexas garantidas — 1 chamada LLM, sem few-shot de exemplos de SQL do seu schema específico (poderia ajudar, mas é iteração futura).

## 5. Ordem de implementação

1. Escolher banco (decisão do usuário, ver seção 2).
2. `pkg/sql/sql.go` — `Config` (DSN), `New` (conecta + introspecção de schema).
3. Prompt de geração de SQL (schema + pergunta → SQL).
4. Validação de segurança (só `SELECT`, sem múltiplos statements) — **antes** de qualquer execução.
5. `Query` — gera, valida, executa, devolve linhas + SQL.
6. Teste da validação de segurança com SQL malicioso fake (`DROP TABLE`, `SELECT ...; DELETE ...`) — confirma rejeição, sem precisar de banco real rodando.
7. Teste de geração com completion fake (schema fixo, pergunta fixa, SQL esperado fixo).
8. Teste manual ponta a ponta: banco real (ex: SQLite/Postgres local com 1-2 tabelas de exemplo), pergunta simples tipo "quantos X existem".

Critério de pronto: `sql.New` + `Query` funcionando ponta a ponta contra um banco real, com a validação de segurança comprovadamente rejeitando SQL destrutivo antes de qualquer execução.

## 6. Decisões tomadas

1. **Banco**: SQLite via `modernc.org/sqlite` (pure Go, sem CGO — mesmo cuidado aprendido com o ANN). Escolhido por não exigir servidor externo rodando, alinhado com o objetivo de binário único embeddable.
2. **Modelo**: `granite4:micro-h` (mesmo do Router/Tree) — funcionou para geração estrutural de SQL (`COUNT`, `ORDER BY ... LIMIT`), sem trocar por modelo maior.

## 7. Status: completo, com limitação real documentada

Implementado em `pkg/sql/` — `New`/`newStoreWithGenerator`, introspecção de schema via `sqlite_master`/`PRAGMA table_info`, `validateReadOnlySelect` (só `SELECT`/`WITH`, 1 statement, sem palavras-chave de escrita/DDL) rodando **antes** de qualquer execução. 8 testes automatizados, incluindo prova de que um `DROP TABLE` gerado é rejeitado e a tabela permanece intacta (`TestStoreQueryRejectsUnsafeGeneratedSQL`).

**Teste manual ponta a ponta** (Ollama real + SQLite real, tabela `funcionarios`):
- "qual o nome do funcionario com o maior salario?" → `SELECT nome FROM funcionarios ORDER BY salario DESC LIMIT 1;` → **`Carla`, correto**.
- "quantos funcionarios sao engenheiros?" → `SELECT COUNT(*) FROM funcionarios WHERE cargo = 'engenheiro';` → **`0`, errado** (deveria ser 2). O banco tem `cargo = 'engenheira'` (forma feminina); o LLM só vê nomes/tipos de coluna na introspecção de schema, não os valores reais, e "chutou" uma forma plausível que não bate com o dado. Não é bug de código — é limitação conhecida de Text-to-SQL baseado só em schema, sem amostra de valores.
- Melhoria possível pra v2 (fora de escopo agora): incluir valores distintos de amostra de colunas de texto na introspecção de schema, pra ancorar o LLM nos valores reais — decisão de design nova (quantos valores, quais colunas, custo de introspecção), não trivial o bastante pra entrar sem uma passada de escopo própria.

---

# Fase 4 — Router com 4 stores (Tree + SQL integrados)

## 1. Escopo

`pkg/router` cobria só vector/graph desde a Fase 2 — na época, Tree e SQL nem existiam ("fora de escopo até existirem", registrado ali mesmo). Agora existem: o Router passa a orquestrar as 4 stores.

- `Strategy` ganha `StrategyTree` e `StrategySQL`.
- `Config` ganha `Tree tree.Config` e `SQL sql.Config` — **as 4 são obrigatórias neste MVP**, sem configuração parcial. Use os pacotes individualmente se não precisar do roteamento completo.
- `QueryResult` ganha `Tree []tree.Node`, `SQLRows []map[string]any`, `SQLQuery string`.
- `AddDocuments` continua fazendo fan-out só pra vector+graph (formato `[]rag.Document`). Tree usa um formato diferente (markdown bruto de um documento inteiro, não trechos pré-cortados) — por isso ganhou um método novo, `AddMarkdownDocument(ctx, id, title, rawMarkdown)`. SQL não tem ingestão via Router — conecta a um banco já populado (introspecção de schema acontece em `sql.New`).
- `classifyPrompt` cresceu de 3 pra 5 categorias, com 1 exemplo novo pra `tree` e 1 pra `sql`.

## 2. Contratos Go (mudanças)

```go
// pkg/router/router.go
type Strategy string
const (
    StrategyVector Strategy = "vector"
    StrategyGraph  Strategy = "graph"
    StrategyBoth   Strategy = "both"
    StrategyTree   Strategy = "tree"
    StrategySQL    Strategy = "sql"
)

type Config struct {
    Vector rag.Config
    Graph  graph.Config
    Tree   tree.Config
    SQL    sql.Config
    RouterModel string
}

type QueryResult struct {
    Strategy Strategy
    Vector   []rag.SearchResult
    Graph    []graph.Relation
    Tree     []tree.Node
    SQLRows  []map[string]any
    SQLQuery string
}

func (r *Router) AddMarkdownDocument(ctx context.Context, id, title, rawMarkdown string) error
```

Pré-requisito de infraestrutura: `pkg/tree` e `pkg/sql` ganharam construtores exportados de injeção de dependência (`tree.NewWithNavigator`, `sql.NewWithGenerator`), mesmo padrão já usado em `rag.NewWithEmbedding`/`graph.NewWithCompletion` — necessários pros testes do Router conseguirem montar fakes das 4 stores.

## 3. Bug real encontrado e corrigido durante a validação ponta a ponta

E2E com as 4 stores reais (Ollama + SQLite): a pergunta "o que o manual diz sobre o motor?" classificou certo como `tree`, mas `Query` voltou **vazio** — não era problema de classificação, era bug em `pkg/tree`:

1. O Ollama respondeu `"1,"` (vírgula sobrando no final) pro estágio de navegação. `parseChosenIndices` fazia `strconv.Atoi` em cada parte separada por vírgula, incluindo a string vazia depois da última vírgula — falhava e descartava a escolha inteira, mesmo tendo entendido "1" corretamente.
2. Isso empurrava pro `keywordFallback`, que também falhava: a pergunta tem `"motor?"` (interrogação colada na palavra), e `strings.Contains(haystack, "motor?")` não bate com `"motor"` no texto real — faltava normalizar pontuação antes de comparar.

Duas correções em `pkg/tree/tree.go`:
- `parseChosenIndices` agora ignora partes vazias (tolera vírgula sobrando) em vez de descartar a resposta inteira.
- `keywordFallback` remove pontuação (`.,!?;:()"'`) de cada palavra da pergunta antes de comparar.

2 testes de regressão novos em `pkg/tree/tree_test.go` (`TestKeywordFallbackStripsPunctuationFromQuestionWords`, `TestParseChosenIndicesToleratesTrailingComma`), sem depender de Ollama.

## 4. Status da classificação em 5 categorias

Validado ponta a ponta com as 4 stores reais, 4 perguntas (1 por categoria principal, sem testar `both` de novo — já documentado como não confiável):

| Pergunta | Esperado | Obtido | Resultado |
|---|---|---|---|
| "quem trabalha na Ignus?" | graph | graph | ✅ (2 relações corretas) |
| "em que pais fica a empresa onde Maria trabalha?" | graph | graph | ✅ |
| "o que o manual diz sobre o motor?" | tree | tree | ✅ (depois da correção do bug acima) |
| "quantos funcionarios sao engenheiros?" | sql | **vector** | ❌ |

A classificação `sql` não disparou — confirmado via diagnóstico direto (curl) que é uma miss real do modelo pequeno (`granite4:micro-h`, 3B), não bug de parsing: a resposta veio `"vector"` de forma limpa. Discriminação em 5 categorias é objetivamente mais difícil que em 3 — consistente com a limitação já documentada pra `both`. Não vale mais engenharia de prompt nesta passada; registrado como limitação conhecida do modelo default, mesma categoria do problema de `both`.

Critério de pronto: dispatch correto testado (fakes) para as 5 estratégias + validação real confirmando 3 de 4 categorias funcionando ponta a ponta com o modelo default, com a 4ª (`sql`) documentada como limitação de modelo, não de código.

---

# Fase 6 — Interface MCP

## 1. Decisões (tomadas com o usuário nesta sessão)

- **1 tool MCP por store** (vector/graph/tree/sql), não 1 tool única via `pkg/router`. Motivo: o agente que fala MCP escolhe a estratégia explicitamente, mais controle e transparência — evita depender da classificação LLM do Router, que já tem limitações documentadas (`both`/`sql` não confiáveis no modelo default).
- **Servidor com estado persistente em memória** — processo MCP fica de pé; `add` numa chamada e `query` noutra operam sobre os mesmos dados carregados no processo. Necessário pro caso de uso real (agente adiciona documento, depois pergunta).
- **SDK oficial**: `github.com/modelcontextprotocol/go-sdk` (mantido pela Anthropic + Google, `mcp.NewServer`/`mcp.AddTool`/`mcp.StdioTransport`). Pure Go, sem CGO. Exige `go >= 1.25` — `go.mod` já está em `go 1.25.0`, sem bump necessário.
- **Transporte**: stdio (`mcp.StdioTransport`) — padrão pra MCP local, sem servidor de rede.

## 2. Escopo dentro do MVP

Novo pacote `pkg/mcpserver/`, que instancia as 4 stores já existentes (`rag.UnifiedRAG`, `graph.GraphStore`, `tree.Tree`, `sql.Store`) uma vez no start do processo e expõe 7 tools:

| Tool | Store | Equivale a |
|---|---|---|
| `vector_add` | `rag.UnifiedRAG.AddDocuments` | `urag add` |
| `vector_query` | `rag.UnifiedRAG.QueryFiltered` | `urag query` |
| `graph_add` | `graph.GraphStore.AddDocuments` | `urag graph ask` (parte add) |
| `graph_query` | `graph.GraphStore.Query` | `urag graph ask` (parte query) |
| `tree_add` | `tree.Tree.AddDocument` | `urag tree ask` (parte add) |
| `tree_query` | `tree.Tree.Query` | `urag tree ask` (parte query) |
| `sql_query` | `sql.Store.Query` | `urag sql query` |

Graph/Tree ganham tools de `add` separadas de `query` (diferente do CLI, que é single-shot) — é exatamente o ganho de ter estado persistente: popular numa chamada MCP, perguntar em várias chamadas seguintes. Vector já tinha essa separação (persistência em disco); aqui a persistência é só em memória (mesma limitação já documentada — não é regressão, Graph/Tree nunca persistiram em disco).

SQL não ganha tool de "add" — mesma decisão já tomada no Router: conecta a um banco já populado, introspecção acontece na config inicial (`-sql-dsn` no start do servidor). Se `-sql-dsn` não for passado, a tool `sql_query` não é registrada (evita expor uma tool que sempre erraria por falta de config).

## 3. Contratos Go

```go
// pkg/mcpserver/server.go
package mcpserver

type Config struct {
    VectorDBPath string // persistência do vector store (chromem-go); "" = in-memory
    LLMModel     string // usado por graph/tree/sql (extração/navegação/geração)
    SQLDSN       string // "" = tool sql_query não é registrada
}

func New(cfg Config) (*Server, error)
func (s *Server) Run(ctx context.Context) error // bloqueante, serve sobre stdio
```

Cada tool segue o padrão do SDK: struct de Input com tags `json`+`jsonschema`, handler `func(ctx, *mcp.CallToolRequest, Input) (*mcp.CallToolResult, Output, error)`, registrado via `mcp.AddTool`.

```go
type VectorAddInput struct {
    Documents []DocumentInput `json:"documents" jsonschema:"documentos a adicionar"`
}
type DocumentInput struct {
    ID      string            `json:"id"`
    Content string            `json:"content"`
    Source  string            `json:"source,omitempty"`
    Meta    map[string]string `json:"meta,omitempty"`
}
type VectorAddOutput struct {
    Added int `json:"added"`
}

type VectorQueryInput struct {
    Question string            `json:"question"`
    TopK     int               `json:"top_k,omitempty"`
    Where    map[string]string `json:"where,omitempty"`
}
type VectorQueryOutput struct {
    Results []SearchResultOutput `json:"results"`
}
```

(`graph_add`/`graph_query`, `tree_add`/`tree_query`, `sql_query` seguem o mesmo padrão, um input/output por tool, campos espelhando os parâmetros que o CLI já expõe por flag.)

## 4. CLI: novo subcomando, não binário novo

`urag mcp serve [-db <path>] [-llm-model <model>] [-sql-dsn <path>]` — reaproveita o binário único já existente (`cmd/urag/main.go`), consistente com o padrão dos outros subcomandos. Sem binário/`cmd/` novo.

## 5. Fora de escopo

- Tool única via Router (decisão tomada, ver seção 1).
- Transporte HTTP/SSE — só stdio.
- Persistência em disco de Graph/Tree via MCP — mesma limitação já documentada nas fases anteriores, não é escopo desta fase resolver.
- Múltiplos servidores MCP simultâneos / multi-tenant — 1 processo, 1 conjunto de stores.
- `sql_add`/ingestão via MCP — SQL conecta a banco já populado, mesma decisão do Router.
- Autenticação/autorização no transporte MCP — stdio local, mesmo modelo de confiança do CLI.

## 6. Ordem de implementação

1. `go get github.com/modelcontextprotocol/go-sdk` (`go.mod` já em `go 1.25.0`, sem bump).
2. `pkg/mcpserver/server.go` — `Config`, `Server`, `New` (instancia as 4 stores, registra tools condicionalmente — `sql_query` só se `SQLDSN != ""`).
3. `pkg/mcpserver/tools.go` — os 7 handlers + tipos de Input/Output.
4. `cmd/urag/main.go` — subcomando `mcp serve`.
5. Teste: cada handler testado diretamente (chamando a função Go, sem subir o transporte stdio) contra stores construídas com fakes (`rag.NewWithEmbedding`, `graph.NewWithCompletion`, etc. — mesmo padrão já usado em todos os outros pacotes).
6. Teste manual ponta a ponta: rodar `urag mcp serve` e trocar mensagens JSON-RPC via stdin/stdout (ou usar um cliente MCP simples) confirmando `vector_add` → `vector_query` no mesmo processo devolve o documento adicionado.

Critério de pronto: build/vet/test limpos incluindo `pkg/mcpserver`, as 7 tools registradas corretamente (6 sempre, `sql_query` condicional), e 1 validação manual real do fluxo add→query dentro da mesma sessão de processo (prova de que o estado persiste em memória entre chamadas, o ganho central desta fase).

## 7. Status: completo, validado com cliente MCP real

Implementado exatamente conforme a seção 3 (`pkg/mcpserver/server.go` + `tools.go`), subcomando `urag mcp serve [-db] [-llm-model] [-sql-dsn]` em `cmd/urag/main.go`. 8 testes automatizados em `tools_test.go`, cobrindo os 7 handlers + o registro condicional de `sql_query` (fakes de embedding/completion/navigate/generate, mesmo padrão de todo o resto do projeto; SQL usa arquivo SQLite real em `t.TempDir()`, não `:memory:`, porque `:memory:` isolado por conexão faria a introspecção do `Store` enxergar um banco vazio diferente do populado no teste).

**Gap descoberto durante os testes**: `sql.Store` nunca expôs um jeito de fechar a conexão (`db.Close()`). Sem isso, o teste com arquivo real travava a limpeza do `t.TempDir()` no Windows ("arquivo já está sendo usado por outro processo"). Corrigido com um `Close() error` mínimo em `sql.Store`, reaproveitado em `mcpserver.Server.Close()` (fecha o sql store se configurado) e chamado via `defer` no subcomando `mcp serve`.

**Validação e2e real**: cliente MCP standalone (`mcp.NewClient` + `mcp.CommandTransport`, executando o binário `urag mcp serve` de verdade) via `go run` num script fora do módulo. Sem Ollama rodando (não precisa: `vector_add`/`vector_query` não chamam LLM). Resultado:
- `tools/list` devolveu exatamente `graph_add`, `graph_query`, `tree_add`, `tree_query`, `vector_add`, `vector_query` — `sql_query` corretamente ausente (sem `-sql-dsn`).
- `vector_add` (2 documentos) → `vector_query("o que os gatos fazem?", top_k=1)` no mesmo processo devolveu `doc1` ("gatos gostam de dormir o dia todo") com score 0.72 — prova real de que o estado persiste em memória entre chamadas MCP separadas, o ganho central desta fase.

---

# Fase 7 — Persistência configurável no MCP + Provider OpenAI-compatível (Graph/Tree/SQL)

## 1. `urag mcp serve`: persistência por padrão

Decisão do usuário: `-db` no `mcp serve` deixa de default para in-memory e passa a default para `./urag_mcp.db` (mesmo espírito do `add`/`query`, que já default para `./urag.db`). Continua ajustável — `-db <path>` aponta pra outro lugar, `-db ""` volta a in-memory. Validado com 2 processos `urag mcp serve` separados: processo 1 fez `vector_add`, processo 2 (novo, sem add) achou o documento via `vector_query` — persistência real confirmada entre execuções, não só entre chamadas do mesmo processo.

`mcp serve` também ganhou `-embedding-provider`/`-embedding-model`/`-embedding-api-key`, ligando no suporte OpenAI que `pkg/rag` já tinha pronto (só não estava exposto no CLI).

## 2. Provider OpenAI-compatível em Graph/Tree/SQL

Antes desta fase, `pkg/graph`, `pkg/tree` e `pkg/sql` só aceitavam `Config.LLMProvider = "ollama"` — qualquer outro valor era erro. Decisão do usuário: suportar também `"openai"`, cobrindo não só a OpenAI oficial mas qualquer provider que implemente o mesmo formato de API (Chat Completions) — vLLM, LM Studio, Together, Groq, etc — via `LLMBaseURL` configurável.

**Novo pacote `internal/openai/complete.go`** — mesmo padrão de `internal/ollama/complete.go` (HTTP direto, sem SDK novo): `Complete(ctx, baseURL, apiKey, model, prompt string, jsonFormat bool) (string, error)`, batendo em `POST {baseURL}/chat/completions` (default `https://api.openai.com/v1`). `jsonFormat` pede `response_format: {type: json_object}` (usado pela extração do Graph RAG, que precisa de JSON estrito).

**`Config` dos 3 pacotes** ganhou `LLMBaseURL` e `LLMAPIKey` (além de `LLMProvider` continuar aceitando valores novos). `New()` em cada um virou um `switch cfg.LLMProvider { case "ollama": ...; case "openai": ...; default: erro }` — sem duplicar a lógica de resolução entre os 3 pacotes (cada um mantém seu próprio `completionFunc`/`navigateFunc`/`generateFunc`, só troca qual `internal/*` cada implementação chama).

`pkg/mcpserver.Config` e todos os subcomandos CLI (`graph ask`, `tree ask`, `sql query`, `router ask`, `mcp serve`) ganharam `-llm-provider`/`-llm-base-url`/`-llm-api-key` (ou os campos equivalentes em `Config`), repassando pra `graph.Config`/`tree.Config`/`sql.Config`.

## 3. Fora de escopo

- **`pkg/router`**: a classificação de estratégia (`classifyStrategy`) continua hardcoded em Ollama — usuário pediu provider externo pra Graph/Tree/SQL, não pro classificador do Router. `router.Config.Graph`/`.Tree`/`.SQL` já herdam a nova opção de provider automaticamente (são os mesmos `graph.Config`/`tree.Config`/`sql.Config`), só a chamada de classificação em si (`ollama.Complete` direto em `router.go`) não foi tocada.
- Outros providers além de OpenAI-compatível (ex: Anthropic, Gemini nativos) — formato de API diferente, fora do que foi pedido.
- Streaming de resposta — `Complete` sempre pede resposta completa de uma vez, mesmo padrão do `internal/ollama`.

## 4. Status: completo, testado com servidor HTTP fake

`internal/openai/complete_test.go`: 3 testes (parse de `choices[0].message.content`, `response_format` correto quando `jsonFormat=true`, erro propagado em status HTTP não-200) via `httptest.NewServer`.

`pkg/graph/provider_test.go`, `pkg/tree/provider_test.go`, `pkg/sql/provider_test.go`: cada um prova, via `httptest.NewServer` simulando o formato OpenAI, que `New(Config{LLMProvider: "openai", ...})` bate em `/chat/completions` (não `/api/generate` do Ollama) e usa a resposta corretamente (extração de entidades no Graph, navegação no Tree, geração de SQL no SQL) — mais um teste de erro pra provider desconhecido em cada pacote.

Build/vet/test limpos em todos os 8 pacotes.

---

# Fase 8 — Transporte HTTP/SSE no servidor MCP

## 1. Motivação

Até aqui `urag mcp serve` só falava **stdio** — funciona bem pra clientes MCP
locais que sobem o binário como subprocesso (Claude Desktop, Claude Code),
mas não é alcançável por uma UI web rodando no navegador (stdio não existe
pra um processo de browser). Usuário levantou a ideia de ter uma tela
separada (outro repo) conectando nesse servidor — decisão: manter a UI num
repo à parte (stack diferente, Go fica sem dependência de frontend), mas
antes disso o uRag-go precisa de um transporte alcançável por rede.

## 2. Decisão: Streamable HTTP (SDK oficial já suporta)

O SDK `modelcontextprotocol/go-sdk` já implementa o transporte "Streamable
HTTP" da spec MCP (`mcp.NewStreamableHTTPHandler` — usa Server-Sent Events
por baixo pra streaming, um único endpoint HTTP). Sem SDK novo, sem
reimplementar handshake — só um `http.Handler` a mais em cima do mesmo
`*mcp.Server` que já existia pro stdio.

Alternativa descartada: `mcp.NewSSEHandler` (transporte SSE "clássico",
anterior à revisão da spec) — Streamable HTTP é a recomendação atual e o que
a maioria dos clientes modernos fala.

## 3. Escopo

- `Server.RunHTTP(ctx, addr string) error` — novo método em
  `pkg/mcpserver/server.go`, ao lado do `Run(ctx)` (stdio) já existente. Usa
  `mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return
  s.mcp }, nil)` — todas as sessões HTTP compartilham a mesma instância de
  `Server` (mesmas stores/estado em memória); sem isolamento multi-tenant,
  mesma decisão de escopo já registrada pro stdio.
- CLI: `urag mcp serve` ganhou `-transport stdio|http` (default `stdio`,
  compatível com o uso anterior) e `-http-addr` (default `:8080`).
- `docker-compose.yml`: o serviço `urag-mcp` passou a rodar com `-transport
  http -http-addr :8080`, porta publicada — faz mais sentido como serviço de
  longa duração num compose (stdio não combina bem com container detached,
  não tem quem segure o stdin).

## 4. Fora de escopo

- Autenticação/autorização no transporte HTTP — mesma postura do resto do
  projeto (stdio também não autentica, é implícito confiar em quem sobe o
  processo). Documentado no README: se for expor além de `localhost`, colocar
  atrás de proxy/gateway que autentique.
- A UI web em si — fica pra outro repositório, decisão do usuário.
- Multi-tenancy / isolamento de sessão HTTP — todas as conexões HTTP
  compartilham as mesmas stores em memória, mesma limitação do stdio.
- Graceful shutdown sofisticado (sinal SIGTERM etc) — `RunHTTP` só reage a
  `ctx.Done()`; não foi adicionado tratamento de sinal do SO nesta fase.

## 5. Status: completo, validado com cliente MCP HTTP real

Validação e2e: `urag mcp serve -db "" -transport http -http-addr
127.0.0.1:18080` rodando em background, cliente MCP standalone usando
`mcp.StreamableClientTransport` (mesmo SDK, lado cliente) conectando em
`http://127.0.0.1:18080`. Resultado: `tools/list` retornou as 6 tools
esperadas (sem `sql_query`, sem `-sql-dsn`); `vector_add` (1 doc) →
`vector_query` no mesmo processo devolveu o doc certo com score 0.72 —
mesma prova de estado em memória já feita pro stdio, agora também via HTTP.

`docker compose config` validado (sintaxe/resolução de volumes e rede ok) —
`docker build`/`up` reais não foram executados (Docker Desktop não estava
disponível no ambiente desta sessão).
