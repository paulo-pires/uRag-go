# Build stage — CGO_ENABLED=0: todas as dependências do projeto são pure-Go
# de propósito (ver SPEC.md, seção "Fase 2 — ANN (HNSW)" e "Fase 3 — Text-to-SQL"
# pra história de por que CGO foi evitado). Isso permite compilar contra
# uma base mínima sem toolchain C.
FROM golang:1.25-alpine AS build
WORKDIR /src

COPY go.mod go.sum ./
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
