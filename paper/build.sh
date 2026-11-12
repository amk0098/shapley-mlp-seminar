#!/usr/bin/env bash
# Builds paper.pdf from the Markdown chapters (requires pandoc >= 3 and xelatex).
set -euo pipefail
cd "$(dirname "$0")"
pandoc 0*.md 1*.md \
  --from markdown+tex_math_dollars \
  --pdf-engine=xelatex \
  --resource-path=. \
  -V documentclass=article -V papersize=a4 -V geometry:margin=2.4cm \
  -V mainfont="DejaVu Serif" -V sansfont="DejaVu Sans" -V monofont="DejaVu Sans Mono" -V fontsize=10pt \
  -V linkcolor=blue -V urlcolor=blue -V colorlinks=true \
  --include-in-header=header.tex \
  --toc --toc-depth=2 \
  -o paper.pdf
echo "written: paper/paper.pdf"
