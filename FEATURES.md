# uRag-go — Funcionalidades

Núcleo do ecossistema uRag: quatro estratégias de RAG unificadas atrás de um Router, expostas como CLI ou servidor MCP.

## Stores de recuperação

- **Vector RAG** (`pkg/rag`) — busca por similaridade semântica via `chromem-go`, com HNSW opcional (`internal/hnsw`, fork vendorizado). Persistência real em disco. Suporta filtro de metadata (`-where`).
- **Graph RAG** (`pkg/graph`) — extrai entidades/relações via LLM e navega multi-hop (BFS). In-memory (sem persistência), decisão deliberada.
- **Vectorless RAG / Tree** (`pkg/tree`) — parseia markdown em árvore de headings e navega via LLM em 2+ estágios, sem embeddings. Pensado pra documentos estruturados (manuais, leis).
- **Text-to-SQL** (`pkg/sql`) — gera SQL a partir de pergunta em linguagem natural contra SQLite (`modernc.org/sqlite`, pure Go); valida que só `SELECT`/`WITH` sejam executados.

## Router

`pkg/router` classifica a pergunta via LLM em `vector`/`graph`/`both`/`tree`/`sql` e despacha pra store certa. Suporta fusão RRF (Reciprocal Rank Fusion) e re-ranking via LLM quando a categoria é `both`. Limitação conhecida: classificação não confiável em modelos pequenos (viés pra `graph`/`vector`).

## Servidor MCP

`urag mcp serve` expõe 8 tools (`vector_add`/`vector_query`, `graph_add`/`graph_query`, `tree_add`/`tree_query`, `sql_query`/`sql_load`) via SDK oficial (`github.com/modelcontextprotocol/go-sdk`). Dois transportes: `stdio` (cliente local, ex: Claude Desktop) e `http` (Streamable HTTP/SSE, pra UI web ou qualquer cliente de rede). Estado persistente em memória entre chamadas — vector store também persiste em disco (`./urag_mcp.db` por padrão).

## CLI

`add`/`query` (Vector RAG, persistente), `graph ask`/`tree ask`/`sql query`/`router ask` (single-shot, in-memory), `mcp serve`.

## Providers de LLM/embedding

Ollama local (default) ou qualquer provider OpenAI-compatível (OpenAI oficial, vLLM, LM Studio, Together, Groq) via `-llm-provider`/`-llm-base-url`/`-llm-api-key`. Embedding configurável separadamente. Observação: a classificação do Router continua sempre via Ollama, independente do provider configurado nas stores.

## Observabilidade

`internal/openai/complete.go` e o cliente Ollama enviam header `X-Urag-Source: urag-go` em toda chamada — usado pelo `uRag-guard-go` pra filtrar guardrails/evals por origem.

## Deploy

`Dockerfile` (multi-stage, `CGO_ENABLED=0`) + `docker-compose.yml` (Ollama + servidor MCP juntos).
