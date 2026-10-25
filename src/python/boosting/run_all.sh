#!/usr/bin/env bash
set -e; cd "$(dirname "$0")"; mkdir -p outputs
for f in 01_adaboost 02_gb_do_zero 03_tabela_comparacao; do python3 "$f.py" | tee "outputs/$f.txt"; done
