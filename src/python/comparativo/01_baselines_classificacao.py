"""Iris: arvores x baselines (regressao logistica, KNN, LDA, SVM), 100 divisoes.
KNN, SVM e regressao logistica usam padronizacao dentro do pipeline (ajustada so no treino)."""
from pathlib import Path
import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
modelos = {
    "Regressão logística": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
    "KNN ($k=5$)":         lambda: make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "LDA":                 lambda: LinearDiscriminantAnalysis(),
    "SVM (RBF)":           lambda: make_pipeline(StandardScaler(), SVC()),
    "Árvore (prof.\\ 3)":   lambda: DecisionTreeClassifier(max_depth=3, random_state=0),
    "Árvore (completa)":   lambda: DecisionTreeClassifier(random_state=0),
}
res = {n: {"acc": [], "f1": []} for n in modelos}
for semente in range(100):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y,
                                              random_state=semente)
    for nome, fabrica in modelos.items():
        pred = fabrica().fit(X_tr, y_tr).predict(X_te)
        res[nome]["acc"].append(np.mean(pred == y_te))
        res[nome]["f1"].append(f1_score(y_te, pred, average="macro"))

virg = lambda v: f"{v:.3f}".replace(".", ",")
linhas = []
for nome, r in res.items():
    print(f"{nome:22s} acc = {np.mean(r['acc']):.3f} ± {np.std(r['acc']):.3f}   "
          f"F1 = {np.mean(r['f1']):.3f} ± {np.std(r['f1']):.3f}")
    linhas.append(f"{nome} & {virg(np.mean(r['acc']))} $\\pm$ {virg(np.std(r['acc']))}"
                  f" & {virg(np.mean(r['f1']))} $\\pm$ {virg(np.std(r['f1']))} \\\\")
tex = ("\\begin{tabular}{@{}lcc@{}}\n\\toprule\n"
       "Método & Acurácia & F1 macro \\\\\n\\midrule\n" + "\n".join(linhas) +
       "\n\\bottomrule\n\\end{tabular}\n")
(Path(__file__).resolve().parent / "outputs" / "tabela_baselines_classificacao.tex").write_text(tex, encoding="utf-8")
