"""Sensibilidade a escala: Iris original x primeira variavel multiplicada por 1000
(por exemplo, trocar cm por micrometros). Nenhum modelo padroniza os dados."""
from pathlib import Path
import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
X_grande = X.copy()
X_grande[:, 0] *= 1000                           # comprimento da sepala x 1000
modelos = {
    "KNN":                 lambda: KNeighborsClassifier(),
    "SVM (RBF)":           lambda: SVC(),
    "Regressão logística": lambda: LogisticRegression(max_iter=5000),
    "LDA":                 lambda: LinearDiscriminantAnalysis(),
    "Árvore (prof.\\ 3)":   lambda: DecisionTreeClassifier(max_depth=3, random_state=0),
}
acc = {n: {"orig": [], "x1000": []} for n in modelos}
for semente in range(50):
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.25,
                                      stratify=y, random_state=semente)
    for nome, fabrica in modelos.items():
        for chave, M in (("orig", X), ("x1000", X_grande)):
            acc[nome][chave].append(fabrica().fit(M[idx_tr], y[idx_tr]).score(M[idx_te], y[idx_te]))

virg = lambda v: f"{v:.3f}".replace(".", ",")
linhas = []
for nome, r in acc.items():
    a, b = np.mean(r["orig"]), np.mean(r["x1000"])
    print(f"{nome:22s} original = {a:.3f}   x1000 = {b:.3f}")
    linhas.append(f"{nome} & {virg(a)} & {virg(b)} \\\\")
tex = ("\\begin{tabular}{@{}lcc@{}}\n\\toprule\n"
       "Método & Original & Variável 1 $\\times\\,1000$ \\\\\n\\midrule\n" + "\n".join(linhas) +
       "\n\\bottomrule\n\\end{tabular}\n")
(Path(__file__).resolve().parent / "outputs" / "tabela_escala.tex").write_text(tex, encoding="utf-8")
