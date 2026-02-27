# sage-rag-benchmark Copilot Instructions

## Scope
- Benchmark suite for `sage-rag` (isage-rag) RAG pipeline components.
- Layer: **L3 · bench** — benchmark, depends on `isage-rag` and lower layers only.

## Polyrepo Context (Important)
SAGE was restructured from a monorepo into a polyrepo. This repo is the dedicated benchmark for `sage-rag`. It depends on the published `isage-rag` package (or local editable install) and should not contain RAG algorithm implementation code.

## Critical rules
- Do not implement RAG components here; they belong in `sage-rag`.
- Do not create new local virtual environments (`venv`/`.venv`); use the existing configured Python environment.
- No fallback logic; fail fast.

## Workflow
1. Add benchmarks in `benchmarks/` or `tests/`.
2. Keep experiment configs declarative (YAML/TOML).
3. Use `sage-rag` as a dependency; do not copy its code.

## Git Hooks (Mandatory)
- Never use `git commit --no-verify` or `git push --no-verify`.
- If hooks fail, fix the issue first.
