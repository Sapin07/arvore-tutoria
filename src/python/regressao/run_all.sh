#!/usr/bin/env bash
set -e; cd "$(dirname "$0")"; mkdir -p outputs figures
for f in 01_arvore_bmi 02_profundidade 03_validacao_cruzada; do python3 "$f.py" | tee "outputs/$f.txt"; done
python3 fig_degraus.py
