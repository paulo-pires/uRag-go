# Build stage — CGO_ENABLED=0: todas as dependências do projeto são pure-Go
# de propósito (ver SPEC.md, seção "Fase 2 — ANN (HNSW)" e "Fase 3 — Text-to-SQL"
# pra história de por que CGO foi evitado). Isso permite compilar contra
# uma base mínima sem toolchain C.
# Base 1.26 vem do lado local (o go.mod exige `go 1.26.0` + `toolchain go1.26.8`,
# que a 1.25-alpine nao compila). WORKDIR /work/uRag-go vem do lado do master: o
# go.mod tem `replace urag-stack/pkg/glitchtip => ../pkg/glitchtip`, e só nesse
# layout `..` cai em /work/pkg/glitchtip, que e onde o COPY --from=glitchtip poe o
# pacote. Com WORKDIR /src o replace apontaria para /pkg/glitchtip e o build
# quebraria — foi por isso que a resolucao nao pegou um lado inteiro.
FROM golang:1.26-alpine AS build
WORKDIR /work/uRag-go
COPY go.mod go.sum ./
COPY --from=glitchtip . /work/pkg/glitchtip
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /out/urag ./cmd/urag

# Runtime stage — imagem final não tem Go nem toolchain, só o binário.
FROM alpine:3.20
RUN apk add --no-cache ca-certificates
WORKDIR /data
COPY --from=build /out/urag /usr/local/bin/urag

ENTRYPOINT ["urag"]
CMD ["mcp", "serve", "-db", "/data/urag_mcp.db"]
