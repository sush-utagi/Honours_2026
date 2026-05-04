#!/bin/bash
# ============================================================
# compress_slides.sh — Compress the seminar slides PDF without
#                      introducing black-bar transparency artifacts.
#
# Usage:  ./scripts/compress_slides.sh          (from project root)
#         ./scripts/compress_slides.sh 150      (custom DPI)
#
# The script acts on slides-fyp.pdf in the project root and
# writes slides-fyp-compressed.pdf alongside it.
#
# Strategy:
#   1. qpdf pre-processes / linearises the PDF (fixes structure).
#   2. Ghostscript compresses images with transparency-safe settings.
#
# The black-bar problem occurs because Ghostscript re-renders pages
# and converts color spaces, mangling transparency groups. Keeping
# everything in DeviceRGB via -dColorConversionStrategy=/LeaveColorUnchanged
# prevents this.
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INPUT="$PROJECT_ROOT/slides-fyp.pdf"
OUTPUT="$PROJECT_ROOT/slides-fyp-compressed.pdf"
DPI="${1:-200}"
TMPFILE="$(mktemp /tmp/pdf_compress_XXXXXX.pdf)"

if [ ! -f "$INPUT" ]; then
  echo "Error: $INPUT not found." >&2
  exit 1
fi

echo "=== Seminar Slides PDF Compression ==="
echo "Input:  $INPUT ($(du -h "$INPUT" | cut -f1))"
echo "Target DPI: $DPI"
echo ""

# --- Step 1: Pre-process with qpdf to fix structure ---
echo "[1/2] Pre-processing with qpdf..."
qpdf --linearize "$INPUT" "$TMPFILE" 2>/dev/null

# --- Step 2: Ghostscript compression with transparency preservation ---
echo "[2/2] Compressing with Ghostscript (transparency-safe)..."
gs \
  -sDEVICE=pdfwrite \
  -dCompatibilityLevel=1.7 \
  -dNOPAUSE -dQUIET -dBATCH \
  -dPrinted=false \
  -dPDFSETTINGS=/printer \
  -dColorConversionStrategy=/LeaveColorUnchanged \
  -dProcessColorModel=/DeviceRGB \
  -dConvertCMYKImagesToRGB=false \
  -dDownsampleColorImages=true \
  -dColorImageResolution="$DPI" \
  -dDownsampleGrayImages=true \
  -dGrayImageResolution="$DPI" \
  -dDownsampleMonoImages=true \
  -dMonoImageResolution=300 \
  -dAutoFilterColorImages=false \
  -dColorImageFilter=/DCTEncode \
  -dAutoFilterGrayImages=false \
  -dGrayImageFilter=/DCTEncode \
  -dCompressPages=true \
  -dDetectDuplicateImages=true \
  -dOptimize=true \
  -sOutputFile="$OUTPUT" \
  "$TMPFILE"

rm -f "$TMPFILE"

echo ""
echo "Output: $OUTPUT ($(du -h "$OUTPUT" | cut -f1))"
echo "Done."
