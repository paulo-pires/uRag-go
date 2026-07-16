# uRag-go

Uma ferramenta RAG unificada e embedável em Go: **Vector RAG**, **Graph RAG**,
**Vectorless RAG** (navegação hierárquica) e **Text-to-SQL** num único
binário, com CLI e um **servidor MCP** pronto pra conectar num agente. Zero
dependências pesadas, zero CGO — só Go puro.

## Por que este projeto existe

A maioria das ferramentas de RAG força toda pergunta pelo mesmo tipo de
busca (normalmente busca vetorial por similaridade). Isso funciona bem para
perguntas factuais simples, mas quebra em três casos comuns:

- **Perguntas relacionais/multi-hop** ("quem trabalha na empresa que fica em
  tal país?") — busca vetorial não modela relações entre entidades.
- **Documentos estruturados** (manuais, relatórios, leis com capítulos e
  seções) — cortar em chunks pra embedding perde a estrutura hierárquica que
  o documento já tem.
- **Dados tabulares** ("quantos funcionários ganham mais de X?") — não é uma
  pergunta de busca textual, é uma pergunta de banco de dados.

uRag-go nasceu pra unificar as quatro abordagens (Vector, Graph, Vectorless,
SQL) atrás de uma única API, com um **Router** que classifica a pergunta e
escolhe a estratégia certa — em vez de reimplementar a mesma busca vetorial
mais uma vez. É um projeto Go, embedável (importa como biblioteca) ou
standalone (CLI / servidor MCP), pensado pra rodar local com
[Ollama](https://ollama.com) sem depender de nenhuma API paga, mas com
suporte a qualquer provider compatível com a API da OpenAI se você preferir
um modelo hosted.

Motivação técnica adicional: todo o histórico de decisões de arquitetura
(por que HNSW foi vendorizado, por que SQLite via `modernc.org/sqlite` e não
`mattn/go-sqlite3`, por que Graph/Tree são in-memory, etc) está registrado em
[`SPEC.md`](SPEC.md), fase por fase — leia lá se quiser o "porquê" de alguma
escolha específica.

## Índice

- [Arquitetura](#arquitetura)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Rodando local](#rodando-local)
  - [CLI: Vector RAG (add/query)](#cli-vector-rag-addquery)
  - [CLI: Graph RAG](#cli-graph-rag)
  - [CLI: Vectorless RAG (Tree)](#cli-vectorless-rag-tree)
  - [CLI: Text-to-SQL](#cli-text-to-sql)
  - [CLI: Router](#cli-router)
  - [Servidor MCP](#servidor-mcp)
- [Configurando o provider de LLM/embedding](#configurando-o-provider-de-llmembedding)
- [Rodando via Docker](#rodando-via-docker)
- [Usando como biblioteca Go](#usando-como-biblioteca-go)
- [Limitações conhecidas](#limitações-conhecidas)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Testes](#testes)
- [Licença](#licença)

## Arquitetura

```
                    ┌─────────────┐
                    │   Router    │  classifica a pergunta (LLM) e despacha
                    └──────┬──────┘
        ┌──────────┬───────┼────────────┐
        ▼          ▼       ▼            ▼
   ┌────────┐ ┌────────┐ ┌──────┐ ┌────────┐
   │ Vector │ │ Graph  │ │ Tree │ │  SQL   │
   │  RAG   │ │  RAG   │ │(Vec- │ │(Text-  │
   │        │ │        │ │torle-│ │to-SQL) │
   │chromem-│ │extração│ │ss)   │ │SQLite +│
   │go +HNSW│ │+ BFS   │ │navega│ │valida- │
   │opcional│ │multi-  │ │ção   │ │ção     │
   │        │ │hop     │ │LLM   │ │read-   │
   │        │ │        │ │2+    │ │only    │
   │        │ │        │ │estág.│ │        │
   └────────┘ └────────┘ └──────┘ └────────┘
```

| Store | Pacote | Quando usar |
|---|---|---|
| **Vector RAG** | `pkg/rag` | Busca por similaridade semântica — perguntas factuais, um documento resolve. |
| **Graph RAG** | `pkg/graph` | Perguntas sobre relações entre entidades, multi-hop. |
| **Vectorless RAG (Tree)** | `pkg/tree` | Documentos estruturados (manual, relatório, lei) — navegação por capítulo/seção em vez de embeddings. |
| **Text-to-SQL** | `pkg/sql` | Dados tabulares — contagens, médias, "quantos", "qual o maior". |
| **Router** | `pkg/router` | Orquestra as 4 acima: classifica a pergunta via LLM e despacha pra store certa (ou pras duas, em perguntas híbridas). |

Cada store é utilizável isoladamente (importa só `pkg/rag`, por exemplo) ou
em conjunto via `pkg/router`. Todos batem em LLM/embedding via HTTP direto —
sem SDK de terceiros pesado —, suportando **Ollama** (local, default) ou
qualquer provider **OpenAI-compatível** (OpenAI oficial, vLLM, LM Studio,
Together, Groq, etc).

Duas formas de consumir o projeto:
1. **CLI** (`cmd/urag`) — comandos diretos por store, mais o Router.
2. **Servidor MCP** (`urag mcp serve`) — expõe as stores como *tools* MCP
   (Model Context Protocol) pra um agente (Claude Code, Claude Desktop, etc)
   chamar diretamente, com estado persistente em memória entre chamadas.

## Pré-requisitos

- **Go 1.25+** (o projeto usa o SDK oficial de MCP, que exige essa versão).
- **[Ollama](https://ollama.com)** rodando local, com os modelos:
  ```
  ollama pull nomic-embed-text     # embedding (Vector RAG)
  ollama pull granite4:micro-h     # extração/navegação/geração/classificação
  ```
  (Ou, alternativamente, uma API key de um provider OpenAI-compatível — ver
  [Configurando o provider](#configurando-o-provider-de-llmembedding).)
- Nenhum banco externo obrigatório: Vector RAG usa arquivos locais
  (`chromem-go`), Text-to-SQL usa SQLite embutido (`modernc.org/sqlite`,
  pure Go, sem CGO).

## Instalação

```bash
git clone <url-do-seu-fork> uRag-go
cd uRag-go
go build -o urag ./cmd/urag
```

Isso gera um binário único `urag` (ou `urag.exe` no Windows). Sem CGO, sem
toolchain C necessária — `CGO_ENABLED=0 go build ...` funciona.

## Rodando local

### CLI: Vector RAG (add/query)

Único par de comandos com persistência real em disco (os demais são
single-shot — ver [Limitações](#limitações-conhecidas)):

```bash
# adicionar documentos (1 linha de texto = 1 documento)
./urag add -source docs.txt -db ./urag.db

# perguntar
./urag query -q "qual a política de férias?" -db ./urag.db -k 5

# com filtro de metadata
./urag query -q "..." -db ./urag.db -where "source=manual.txt"
```

### CLI: Graph RAG

```bash
./urag graph ask -source docs.txt -q "quem trabalha na Ignus e onde ela fica?" -hops 2
```

Carrega, extrai entidades/relações via LLM e responde na mesma invocação
(in-memory, sem persistência — ver limitações).

### CLI: Vectorless RAG (Tree)

```bash
./urag tree ask -source manual.md -title "Manual do Produto" -q "o que diz sobre manutenção do motor?" -depth 3
```

Parseia o markdown em árvore de headings (`#`, `##`, `###`) e navega via LLM
em 2+ estágios até achar a seção relevante.

### CLI: Text-to-SQL

```bash
./urag sql query -dsn ./dados.db -q "quantos funcionários ganham mais de 5000?"
```

Só `SELECT`/`WITH` são permitidos — qualquer outro tipo de statement gerado
pelo LLM é rejeitado antes de tocar o banco (ver `pkg/sql/sql.go:validateReadOnlySelect`).

### CLI: Router

Orquestra as 4 stores — exige as 4 configuradas (sem modo parcial neste
MVP; use os pacotes individuais se não precisar do roteamento completo):

```bash
./urag router ask \
  -source docs.txt \
  -tree-source manual.md -tree-title "Manual" \
  -sql-dsn ./dados.db \
  -q "quantos funcionários trabalham na Ignus?"
```

O Router classifica a pergunta via LLM em `vector`/`graph`/`both`/`tree`/`sql`
e despacha pra store certa — ver `SPEC.md` (Fase 3/4) pra limitações
conhecidas de classificação em modelos pequenos.

### Servidor MCP

```bash
# transporte stdio (default) — pra clientes que sobem o binário como subprocesso
./urag mcp serve -db ./urag_mcp.db -sql-dsn ./dados.db

# transporte HTTP/SSE — pra uma UI web ou qualquer cliente na rede
./urag mcp serve -db ./urag_mcp.db -sql-dsn ./dados.db -transport http -http-addr :8080
```

Sobe um servidor [MCP](https://modelcontextprotocol.io) com 7 tools
(vector/graph/tree têm `_add` e `_query`; SQL só `sql_query`, e só se
`-sql-dsn` for passado), em dois transportes possíveis via `-transport`:

| Transporte | Quando usar |
|---|---|
| `stdio` (default) | Cliente MCP local que sobe o binário como subprocesso (Claude Desktop, Claude Code) — não escuta rede. |
| `http` | Streamable HTTP/SSE — servidor escuta em `-http-addr` (default `:8080`), qualquer cliente na rede conecta por HTTP (ex: uma UI web separada). |

| Tool | Descrição |
|---|---|
| `vector_add` / `vector_query` | Vector RAG |
| `graph_add` / `graph_query` | Graph RAG |
| `tree_add` / `tree_query` | Vectorless RAG |
| `sql_query` | Text-to-SQL (só se `-sql-dsn` configurado) |

Diferente dos comandos `ask` do CLI (single-shot), o servidor MCP mantém
**estado em memória entre chamadas** — um agente pode chamar `graph_add`
numa invocação e `graph_query` em outra, no mesmo processo. O vector store
persiste em disco por padrão (`./urag_mcp.db`); graph/tree continuam
in-memory (perdem estado se o processo cair — decisão deliberada, ver
`SPEC.md`).

Pra conectar num cliente MCP local (ex: Claude Desktop, Claude Code), aponte
pro binário como comando stdio, por exemplo num `mcpServers` config:

```json
{
  "mcpServers": {
    "urag": {
      "command": "/caminho/absoluto/para/urag",
      "args": ["mcp", "serve", "-db", "/caminho/para/urag_mcp.db", "-sql-dsn", "/caminho/para/dados.db"]
    }
  }
}
```

Pra conectar por **HTTP** (ex: uma UI web num outro repo/servidor), suba com
`-transport http` e aponte o cliente pro endpoint (`http://host:porta`) — o
SDK oficial de MCP em qualquer linguagem já sabe falar esse transporte
("Streamable HTTP", que usa Server-Sent Events por baixo pra streaming). Não
há autenticação embutida hoje — se for expor além de `localhost`, coloque
atrás de um proxy/gateway que autentique antes de repassar.

## Configurando o provider de LLM/embedding

Por padrão tudo roda contra **Ollama local** (`http://localhost:11434`). Pra
usar um provider **OpenAI-compatível** (OpenAI oficial, vLLM, LM Studio,
Together, Groq, etc), use as flags abaixo — disponíveis em `graph ask`,
`tree ask`, `sql query`, `router ask` e `mcp serve`:

| Flag | Descrição |
|---|---|
| `-llm-provider` | `ollama` (default) ou `openai` |
| `-llm-base-url` | override do endpoint — obrigatório pra provider OpenAI-compatível que não seja a OpenAI oficial (ex: `http://localhost:8000/v1` num vLLM local) |
| `-llm-api-key` | API key, obrigatória se `-llm-provider=openai` |

O **embedding** (só usado pelo Vector RAG) é configurado separadamente, hoje
só exposto em `mcp serve` (o `add`/`query` do CLI usam Ollama fixo por
simplicidade):

| Flag | Descrição |
|---|---|
| `-embedding-provider` | `ollama` (default) ou `openai` |
| `-embedding-model` | modelo de embedding (default: `nomic-embed-text` no Ollama) |
| `-embedding-base-url` | override do endpoint Ollama de embedding |
| `-embedding-api-key` | API key, obrigatória se `-embedding-provider=openai` |

Exemplo com OpenAI oficial:

```bash
./urag mcp serve \
  -embedding-provider openai -embedding-model text-embedding-3-small -embedding-api-key sk-... \
  -llm-provider openai -llm-model gpt-4o-mini -llm-api-key sk-...
```

**Nota**: a classificação de estratégia do `router ask` (qual store responde
a pergunta) continua sempre via Ollama — só as stores individuais
(Graph/Tree/SQL) aceitam provider alternativo hoje (ver `SPEC.md`, Fase 7).

## Rodando via Docker

O projeto inclui `Dockerfile` (build multi-stage, `CGO_ENABLED=0`, imagem
final `alpine` mínima) e `docker-compose.yml` (Ollama + o servidor MCP
juntos).

### Só o binário, contra um Ollama já rodando no host

```bash
docker build -t urag .
docker run --rm -it \
  -v "$(pwd)/data:/data" \
  --add-host=host.docker.internal:host-gateway \
  urag mcp serve -db /data/urag_mcp.db \
    -embedding-base-url http://host.docker.internal:11434 \
    -llm-base-url http://host.docker.internal:11434
```

(No Linux, troque `host.docker.internal` pelo IP da interface docker0, ou
use `--network=host`.)

### Com docker-compose (Ollama + urag no mesmo compose)

```bash
docker compose up
```

Isso sobe dois serviços:
- `ollama`: imagem oficial `ollama/ollama`, com volume persistente pros
  modelos baixados.
- `urag-mcp`: builda a partir do `Dockerfile` local, roda `mcp serve
  -transport http -http-addr :8080` (porta publicada em `8080:8080`), aponta
  `-embedding-base-url`/`-llm-base-url` pro serviço `ollama` (nome resolvido
  pela rede interna do compose), persiste o vector store no volume
  `urag_data`.

Antes do primeiro uso, baixe os modelos dentro do container `ollama`:

```bash
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull granite4:micro-h
```

Depois do `up`, o servidor MCP está acessível em `http://localhost:8080` —
qualquer cliente MCP que fale o transporte Streamable HTTP (inclusive uma UI
web num outro repo) conecta direto nessa porta, sem precisar subir o
processo como subprocesso. Pra usar a CLI (comandos `add`/`query`/`graph
ask`/etc, não o servidor MCP) dentro do compose:

```bash
docker compose run --rm urag-mcp add -source /data/docs.txt -db /data/urag.db
```

## Usando como biblioteca Go

Cada pacote é importável isoladamente:

```go
import "urag-go/pkg/rag"

ur, err := rag.New(rag.Config{
    EmbeddingProvider: "ollama",
    EmbeddingModel:    "nomic-embed-text",
    PersistPath:       "./urag.db",
})
if err != nil {
    log.Fatal(err)
}

ur.AddDocuments(ctx, []rag.Document{{ID: "doc1", Content: "..."}})
results, _ := ur.Query(ctx, "pergunta", 5)
```

Mesma coisa pra `pkg/graph`, `pkg/tree`, `pkg/sql` e `pkg/router` — cada um
tem seu próprio `Config`/`New`. Contratos completos em `SPEC.md`, seção
"Contratos Go" de cada fase.

## Limitações conhecidas

Documentadas em detalhe no `SPEC.md` (não são bugs, são decisões
registradas):

1. **Graph e Tree são in-memory, sem persistência em disco** — por isso os
   comandos `ask` do CLI são single-shot (carregam e consultam na mesma
   invocação). O servidor MCP contorna isso mantendo o processo vivo, mas
   ainda perde os dados se o processo cair.
2. **Classificação do Router em 5 categorias** não é totalmente confiável no
   modelo default (`granite4:micro-h`, 3B) — viés pra `graph`/`vector` em
   perguntas ambíguas classificadas como `both`/`sql`. Um modelo maior
   resolveria; não trocamos o default por causa do custo de rodar local.
3. **Text-to-SQL não vê valores de coluna**, só nomes/tipos — pode gerar
   `WHERE cargo = 'engenheiro'` quando o dado real é `'engenheira'`.

## Estrutura do projeto

```
uRag-go/
├── cmd/urag/          # CLI (add/query/graph/tree/sql/router/mcp)
├── pkg/
│   ├── rag/           # Vector RAG (wrapper chromem-go + HNSW opcional)
│   ├── graph/         # Graph RAG (extração + BFS multi-hop)
│   ├── tree/          # Vectorless RAG (parser markdown + navegação LLM)
│   ├── sql/           # Text-to-SQL (SQLite + validação read-only)
│   ├── router/        # Orquestra as 4 stores acima
│   └── mcpserver/     # Servidor MCP (7 tools sobre as 4 stores)
├── internal/
│   ├── ollama/        # Cliente HTTP compartilhado pro Ollama
│   ├── openai/        # Cliente HTTP compartilhado pra providers OpenAI-compatíveis
│   └── hnsw/          # Fork vendorizado de coder/hnsw (ver internal/hnsw/NOTICE.md)
├── SPEC.md            # Histórico completo de decisões, fase por fase
├── Dockerfile
├── docker-compose.yml
└── LICENSE
```

## Testes

```bash
go build ./... && go vet ./... && go test ./...
```

Todos os testes usam fakes (embedding/completion/navigate/generate
determinísticos) — nenhum depende de Ollama ou rede real rodando.

## Licença

[MIT](LICENSE).
