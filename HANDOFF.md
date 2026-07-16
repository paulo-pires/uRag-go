# uRag-go — Handoff pra próxima sessão

> Leia isto primeiro. Detalhe completo de cada decisão está no `SPEC.md` (grande,
> organizado por fase — use como referência, não como leitura corrida).

## Estado atual: tudo verde

```
go build ./... && go vet ./... && go test ./...
```
Passa limpo. 7 pacotes: `pkg/rag` (Vector RAG), `pkg/graph` (Graph RAG), `pkg/tree`
(Vectorless RAG), `pkg/sql` (Text-to-SQL), `pkg/router` (orquestra as 4),
`pkg/mcpserver` (servidor MCP, 7 tools), mais `internal/hnsw` (fork vendorizado
do `coder/hnsw`) e `internal/ollama` (HTTP compartilhado pro Ollama). CLI em
`cmd/urag/main.go`: `add`/`query` (Vector RAG, persistente) + `graph ask`/`tree
ask`/`sql query`/`router ask` (single-shot, in-memory) + `mcp serve` (servidor
MCP com estado persistente em memória).

Tudo do roadmap combinado com o usuário está feito: filtros de metadata, Graph
RAG, ANN/HNSW, Router (com fusão, multi-estratégia, few-shot), Vectorless RAG,
Text-to-SQL, integração das 4 stores no Router, CLI, **e agora Interface MCP**.

## Interface MCP: feita nesta sessão

Escopo decidido com o usuário (registrado no SPEC.md, Fase 6, antes de codar):
1 tool por store (não 1 tool única via Router), servidor com estado
persistente em memória (não single-shot como o CLI), SDK oficial
`github.com/modelcontextprotocol/go-sdk` (Anthropic+Google, pure Go, exige
`go >= 1.25` — `go.mod` já estava em `go 1.25.0`, sem bump necessário),
transporte stdio.

`pkg/mcpserver/` expõe 7 tools: `vector_add`/`vector_query`,
`graph_add`/`graph_query`, `tree_add`/`tree_query`, `sql_query` (só registrada
se `-sql-dsn` for passado — SQL conecta a banco já populado, mesma decisão já
tomada no Router). `urag mcp serve [-db <path>] [-llm-model <model>] [-sql-dsn
<path>]` sobe o servidor.

Validado com um cliente MCP real (`mcp.NewClient` + `mcp.CommandTransport`,
executando o binário de verdade) — `vector_add` numa chamada, `vector_query`
noutra, mesmo processo, resultado correto. Prova real de que o estado
persiste em memória entre chamadas MCP, que era o ganho central pedido.

Gap descoberto e corrigido: `sql.Store` não tinha `Close()` — adicionado
(pequeno, reaproveitado em `mcpserver.Server.Close()`).

## Fase 7 feita nesta sessão: persistência configurável + provider OpenAI

`urag mcp serve` agora persiste por padrão em `./urag_mcp.db` (antes era
in-memory) — validado com 2 processos separados, o segundo achou o
documento adicionado pelo primeiro. `-db` continua ajustável pra outro
caminho (ou `""` pra voltar a in-memory).

`pkg/graph`, `pkg/tree` e `pkg/sql` agora aceitam `LLMProvider = "openai"`
além de `"ollama"` — cobre a OpenAI oficial e qualquer provider que
implemente o mesmo formato de Chat Completions (vLLM, LM Studio, Together,
Groq, etc), via `LLMBaseURL` configurável. Novo `internal/openai/complete.go`
(mesmo padrão HTTP direto de `internal/ollama`). CLI (`graph ask`/`tree
ask`/`sql query`/`router ask`/`mcp serve`) ganhou `-llm-provider`/
`-llm-base-url`/`-llm-api-key`.

**Fora de escopo, registrado no SPEC.md**: a classificação do Router
(`classifyStrategy`) continua hardcoded em Ollama — só as 3 stores
individuais ganharam o provider novo, não o classificador do Router em si.

Testado com `httptest.NewServer` fake (formato OpenAI) em `internal/openai`
e em cada um dos 3 pacotes — sem precisar de API key real ou rede.
Build/vet/test limpos nos 8 pacotes.

## Fase 8 feita nesta sessão: transporte HTTP/SSE no servidor MCP

Também produzido nesta sessão: `README.md` completo (motivação, uso local,
Docker) e `LICENSE` (MIT). `Dockerfile` + `docker-compose.yml` novos.

Contexto: usuário quer eventualmente uma UI web separada (outro
repositório) conectando neste servidor MCP. Decisão: manter a UI num repo à
parte (Go fica sem dependência de frontend), mas primeiro o uRag-go
precisava de um transporte alcançável por rede — stdio só funciona quando o
cliente sobe o processo como subprocesso, não serve pra um browser.

`urag mcp serve` ganhou `-transport stdio|http` (default `stdio`, compatível
com uso anterior) e `-http-addr` (default `:8080`). HTTP usa
`mcp.NewStreamableHTTPHandler` do próprio SDK (Streamable HTTP — usa SSE por
baixo pra streaming) — sem lib nova. `pkg/mcpserver.Server` ganhou
`RunHTTP(ctx, addr)` ao lado do `Run(ctx)` (stdio) já existente.

`docker-compose.yml`: serviço `urag-mcp` passou a rodar em modo HTTP (porta
8080 publicada) — faz mais sentido como serviço de longa duração num
compose do que stdio (que precisa de alguém segurando o stdin).

Validado com um cliente MCP HTTP real (`mcp.StreamableClientTransport`)
contra o binário rodando de verdade em background — `vector_add` →
`vector_query` no mesmo processo, resultado correto, mesma prova de estado
em memória já feita pro stdio.

**Não validado**: `docker build`/`docker compose up` reais — Docker Desktop
não estava disponível neste ambiente. Só `docker compose config` (sintaxe)
foi confirmado. Vale rodar de verdade assim que houver Docker disponível.

## Limitações conhecidas (não são bugs, já documentadas)

1. **Classificação do Router em 5 categorias**: modelo default (`granite4:micro-h`,
   3B) não é confiável pra `both`/`sql` em perguntas ambíguas — viés forte pra
   `graph`/`vector`. Funciona bem pra `vector`/`graph`/`tree` simples. Corrigir
   exigiria modelo maior ou reestruturar o prompt — não tentei mais nesta sessão
   por causa do custo.
2. **Text-to-SQL sem amostra de valores**: o LLM só vê nomes/tipos de coluna na
   introspecção de schema, não os valores reais — pode gerar `WHERE cargo =
   'engenheiro'` quando o dado real é `'engenheira'`. v2 possível: incluir
   valores distintos de amostra no prompt.
3. **Graph e Tree são in-memory, sem persistência** — decisão deliberada, não
   esquecimento. Por isso os comandos de CLI pra eles são single-shot.

## Como validar mudanças (padrão usado a sessão inteira)

- Testes automatizados usam fakes (`fakeEmbedding`, `fakeExtraction`,
  `fakeNavigate`, etc.) — nunca dependem de Ollama rodando.
- Pra e2e real: escrever um `go run` script standalone (fora do módulo, ex: em
  `%TEMP%\claude\...\scratchpad\`) importando os pacotes, chamando `New(...)`
  com provider `"ollama"`, e imprimindo o resultado. `go run <path absoluto>`
  funciona mesmo com o arquivo fora do diretório do módulo, contanto que o
  `cwd` do comando seja a raiz do módulo.
- Modelos Ollama já testados e funcionando: `nomic-embed-text` (embedding),
  `granite4:micro-h` (classificação/extração/navegação/geração — bom
  instruction-following, sem os problemas de `qwen3.5:0.8b` — modelo "thinking",
  resposta cai no campo errado da API — nem de `qwen2.5-coder:3b` — modelo de
  código, não generalista), `hf.co/unsloth/gemma-4-E2B-it-GGUF:IQ4_NL` (usado
  no Graph RAG especificamente, decisão do usuário).

## Armadilhas já resolvidas (não repetir a investigação)

- **CGO quebra o build no Windows silenciosamente às vezes só na hora de
  compilar uma dependência transitiva.** Aconteceu com `coder/hnsw` (via
  `google/renameio` v1, sem suporte a Windows) e quase aconteceu com
  `habedi/hann` (usa CGO direto pra SIMD). Sempre preferir dependências
  pure-Go; testar `go build ./...` logo depois de qualquer `go get` novo.
- **`go get` pode forçar upgrade de toolchain Go** (`GOTOOLCHAIN=auto`) — isso
  é normal/esperado no Go moderno, não precisa reverter `go.mod` por causa
  disso (só reverti uma vez por engano, não repetir).
- Estrutura de pacotes segue um padrão consistente: `New(cfg Config)
  (*X, error)` público + `NewWithY(dependência) *X` ou `(*X, error)` exportado
  pra injeção de dependência (usado em testes cross-package) + `newXWithY`
  não-exportado internamente. Ver `rag.NewWithEmbedding`,
  `graph.NewWithCompletion`, `tree.NewWithNavigator`, `sql.NewWithGenerator`.

## Custo desta sessão

Ficou muito alto (~$97) por causa do volume de validação real com Ollama
(vários e2e por fase, várias fases). Pra próxima sessão: considerar agrupar
mais validações num e2e só por fase, e ser mais seletivo sobre quando vale a
pena re-testar via LLM real vs confiar em testes com fake + análise de código.
