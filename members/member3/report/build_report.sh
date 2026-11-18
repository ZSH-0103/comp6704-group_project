#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p build
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build member3_report.tex
else
  /Library/TeX/texbin/latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build member3_report.tex
fi
cp build/member3_report.pdf member3_report.pdf
