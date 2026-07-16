# Vendored fork of github.com/coder/hnsw v0.6.1

Copiado de https://github.com/coder/hnsw (CC0 1.0, ver LICENSE nesta pasta), com uma
única mudança: `SavedGraph`, `LoadSavedGraph` e `(*SavedGraph).Save` foram removidos
de `encode.go`.

**Motivo**: `Save()` chamava `renameio.TempFile`, e a v1 de `github.com/google/renameio`
não tem implementação de `TempFile` para Windows (arquivo `tempfile.go` da lib tem
`// +build !windows`, sem equivalente Windows) — o pacote não compilava neste SO.

**Por que vendorizar em vez de trocar de lib**: não usamos persistência do índice ANN
(decisão de escopo em SPEC.md — `Index=hnsw` não é combinável com `PersistPath`), então
a função quebrada nunca seria chamada de qualquer forma. `Export`/`Import` (que não
dependem de renameio) continuam disponíveis sem alteração.

Ao atualizar: reaplicar essa remoção se re-vendorizar uma versão nova do upstream.
