#!/usr/bin/env bash
# Build the replication report: render the report figures, then compile the PDF.
# Run from anywhere; it cd's to the repo root. Needs typst + the eval runs under
# data/eval/compare/. F10/F11 render only once the human/Turing panels are rated.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python scripts/plot_report.py
typst compile report/report.typ report/persona-rag-report.pdf
echo "wrote report/persona-rag-report.pdf"
