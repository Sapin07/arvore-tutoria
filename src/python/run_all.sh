#!/usr/bin/env bash
# Regenera TODAS as saidas e figuras usadas pelos slides. Uso: bash src/python/run_all.sh
set -e; cd "$(dirname "$0")"
# usa o .venv criado por setup.sh, se existir (os run_all.sh internos chamam python3)
[ -d ../../.venv ] && export PATH="$(cd ../../.venv/bin && pwd):$PATH"
for d in ensembles regressao boosting poda comparativo; do echo "### $d"; bash "$d/run_all.sh" > /dev/null; done
echo "ok"
