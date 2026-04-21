#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# build_seminar_slides.sh — Compile the Honours seminar slides and copy
#                           the PDF to the project root.
#
# Usage:
#   ./scripts/build_seminar_slides.sh          # full build (default)
#   ./scripts/build_seminar_slides.sh --clean  # remove build artifacts only
#
# Requirements: pdflatex, bibtex (typically via TeX Live / MacTeX)
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Resolve paths ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SEMINAR_DIR="$PROJECT_ROOT/thesis/CSSE-Hons-seminar-template-released"
TEX_FILE="slides-fyp.tex"
PDF_FILE="slides-fyp.pdf"
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
    info "Cleaning build artifacts in $SEMINAR_DIR …"
    cd "$SEMINAR_DIR"
    # Standard LaTeX intermediates + bibtex
    rm -f slides-fyp.{aux,bbl,blg,log,out,toc,nav,snm,vrb,fls,fdb_latexmk,synctex.gz} 2>/dev/null || true
    info "Done."
}

# ── Handle --clean flag ──────────────────────────────────────────
if [[ "${1:-}" == "--clean" ]]; then
    clean
    exit 0
fi

# ── Pre-flight checks ───────────────────────────────────────────
for cmd in pdflatex bibtex; do
    if ! command -v "$cmd" &>/dev/null; then
        error "'$cmd' not found. Install TeX Live / MacTeX first."
        exit 1
    fi
done

if [[ ! -f "$SEMINAR_DIR/$TEX_FILE" ]]; then
    error "Cannot find $SEMINAR_DIR/$TEX_FILE"
    exit 1
fi

# ── Build ────────────────────────────────────────────────────────
cd "$SEMINAR_DIR"
info "Starting seminar slides build …"

# Pass 1 — initial compile (generates .aux for bibtex)
info "  pdflatex pass 1/3 …"
pdflatex -interaction=nonstopmode -halt-on-error "$TEX_FILE" > /dev/null 2>&1 || {
	error "pdflatex pass 1 failed. Run manually in $SEMINAR_DIR to see errors."
	exit 1
}

# Bibtex — resolve bibliography
info "  bibtex …"
bibtex slides-fyp > /dev/null 2>&1 || true

# Pass 2 — incorporate references
info "  pdflatex pass 2/3 …"
pdflatex -interaction=nonstopmode -halt-on-error "$TEX_FILE" > /dev/null 2>&1 || {
	error "pdflatex pass 2 failed."
	exit 1
}

# Pass 3 — finalise cross-references, ToC
info "  pdflatex pass 3/3 …"
pdflatex -interaction=nonstopmode -halt-on-error "$TEX_FILE" > /dev/null 2>&1 || {
	error "pdflatex pass 3 failed."
	exit 1
}

# ── Copy to project root ────────────────────────────────────────
if [[ -f "$PDF_FILE" ]]; then
    cp "$PDF_FILE" "$OUTPUT_PDF"
    info "Slides compiled successfully → ${OUTPUT_PDF}"
else
    error "Build seemed to succeed but $PDF_FILE was not produced."
    exit 1
fi
