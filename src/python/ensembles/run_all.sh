#!/usr/bin/env bash
# Regenera outputs/*.txt e figures/* usados pelos slides da secao de ensembles.
# Uso (da raiz do projeto):  bash src/python/ensembles/run_all.sh
set -e
cd "$(dirname "$0")"
mkdir -p outputs figures
python3 ../export_iris.py
for f in 01_maioria 02_votacao 02b_votacao_repetido 03_bootstrap 04_bagging 04b_bagging_repetido 05_oob; do
  python3 "$f.py" | tee "outputs/$f.txt"
done
for f in fig_maioria fig_fronteiras fig_acuracia_vs_B; do
  python3 "$f.py"
done
python3 -c "import sklearn, numpy, scipy, matplotlib; print('scikit-learn', sklearn.__version__, '| numpy', numpy.__version__, '| scipy', scipy.__version__, '| matplotlib', matplotlib.__version__)" > outputs/versions.txt
