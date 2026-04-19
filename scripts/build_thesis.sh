#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# build_thesis.sh — Compile the Honours thesis and copy the PDF
#                   to the project root.
#
# Usage:
#   ./scripts/build_thesis.sh          # full build (default)
#   ./scripts/build_thesis.sh --clean  # remove build artifacts only
#
# Requirements: pdflatex, biber  (typically via TeX Live / MacTeX)
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Resolve paths ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
THESIS_DIR="$PROJECT_ROOT/thesis/CSSE-Hons-thesis-template-released"
TEX_FILE="thesis.tex"
PDF_FILE="thesis.pdf"
OUTPUT_PDF="$PROJECT_ROOT/$PDF_FILE"

# ── Colours for output ───────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Colour

info()  { echo -e "${GREEN}[build]${NC} $*"; }
warn()  { echo -e "${YELLOW}[build]${NC} $*"; }
error() { echo -e "${RED}[build]${NC} $*" >&2; }

# ── Clean helper ─────────────────────────────────────────────────
clean() {
    info "Cleaning build artifacts in $THESIS_DIR …"
    cd "$THESIS_DIR"
    # Standard LaTeX intermediates + biber
    rm -f thesis.{aux,bbl,bcf,blg,lof,log,lot,out,run.xml,toc,fls,fdb_latexmk,synctex.gz} 2>/dev/null || true
    info "Done."
}

# ── Handle --clean flag ──────────────────────────────────────────
if [[ "${1:-}" == "--clean" ]]; then
    clean
    exit 0
fi

# ── Pre-flight checks ───────────────────────────────────────────
for cmd in pdflatex biber; do
    if ! command -v "$cmd" &>/dev/null; then
        error "'$cmd' not found. Install TeX Live / MacTeX first."
        exit 1
    fi
done

if [[ ! -f "$THESIS_DIR/$TEX_FILE" ]]; then
    error "Cannot find $THESIS_DIR/$TEX_FILE"
    exit 1
fi

# ── Build ────────────────────────────────────────────────────────
cd "$THESIS_DIR"
info "Starting thesis build …"

# Pass 1 — initial compile (generates .bcf for biber)
info "  pdflatex pass 1/3 …"
pdflatex -interaction=nonstopmode -halt-on-error "$TEX_FILE" > /dev/null 2>&1

# Biber — resolve bibliography
info "  biber …"
biber thesis > /dev/null 2>&1

# Pass 2 — incorporate references
info "  pdflatex pass 2/3 …"
pdflatex -interaction=nonstopmode -halt-on-error "$TEX_FILE" > /dev/null 2>&1

# Pass 3 — finalise cross-references, ToC, LoF, LoT
info "  pdflatex pass 3/3 …"
pdflatex -interaction=nonstopmode -halt-on-error "$TEX_FILE" > /dev/null 2>&1

# ── Copy to project root ────────────────────────────────────────
if [[ -f "$PDF_FILE" ]]; then
    cp "$PDF_FILE" "$OUTPUT_PDF"
    info "Thesis compiled successfully → ${OUTPUT_PDF}"
else
    error "Build seemed to succeed but $PDF_FILE was not produced."
    exit 1
fi
