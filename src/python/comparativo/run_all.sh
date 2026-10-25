#!/usr/bin/env bash
set -e; cd "$(dirname "$0")"; mkdir -p outputs figures
for f in 01_baselines_classificacao 02_baselines_regressao 03_escala 04_rotacao; do python3 "$f.py" | tee "outputs/$f.txt"; done
python3 fig_rotacao.py
