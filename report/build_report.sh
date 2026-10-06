#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p build
latexmk -pdf -outdir=build -interaction=nonstopmode -halt-on-error main.tex
cp build/main.pdf main.pdf
