"""Arvore (gini x entropia), bagging, AdaBoost e gradient boosting no Iris:
100 divisoes treino/teste estratificadas; escreve outputs/tabela_comparacao.tex."""
from pathlib import Path
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier)

X, y = load_iris(return_X_y=True)
modelos = {
    "Árvore (Gini)":        lambda: DecisionTreeClassifier(random_state=0),
    "Árvore (entropia)":    lambda: DecisionTreeClassifier(criterion="entropy", random_state=0),
    "Bagging ($B=200$)":    lambda: BaggingClassifier(DecisionTreeClassifier(),
                                                      n_estimators=200, random_state=0),
    "AdaBoost ($B=200$, \\emph{stumps})":
                            lambda: AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                                       n_estimators=200, random_state=0),
    "Gradient boosting ($B=100$, $d=2$)":
                            lambda: GradientBoostingClassifier(n_estimators=100, max_depth=2,
                                                               random_state=0),
}
res = {nome: {"acc": [], "f1": []} for nome in modelos}
for semente in range(100):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              stratify=y, random_state=semente)
    for nome, fabrica in modelos.items():
        pred = fabrica().fit(X_tr, y_tr).predict(X_te)
        res[nome]["acc"].append(np.mean(pred == y_te))
        res[nome]["f1"].append(f1_score(y_te, pred, average="macro"))

virg = lambda v: f"{v:.3f}".replace(".", ",")
linhas = []
for nome, r in res.items():
    print(f"{nome:36s} acc = {np.mean(r['acc']):.3f} ± {np.std(r['acc']):.3f}   "
          f"F1 = {np.mean(r['f1']):.3f} ± {np.std(r['f1']):.3f}")
    linhas.append(f"{nome} & {virg(np.mean(r['acc']))} $\\pm$ {virg(np.std(r['acc']))}"
                  f" & {virg(np.mean(r['f1']))} $\\pm$ {virg(np.std(r['f1']))} \\\\")
tex = ("\\begin{tabular}{@{}lcc@{}}\n\\toprule\n"
       "Método & Acurácia & F1 macro \\\\\n\\midrule\n" + "\n".join(linhas) +
       "\n\\bottomrule\n\\end{tabular}\n")
(Path(__file__).resolve().parent / "outputs" / "tabela_comparacao.tex").write_text(tex, encoding="utf-8")
