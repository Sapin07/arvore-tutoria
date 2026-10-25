#!/usr/bin/env bash
# Cria o ambiente Python do projeto em .venv e instala as dependencias.
# Uso: bash setup.sh   (depois: source .venv/bin/activate)
set -e; cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
echo "ok ambiente em .venv. Para regenerar as saidas: bash src/python/run_all.sh"
