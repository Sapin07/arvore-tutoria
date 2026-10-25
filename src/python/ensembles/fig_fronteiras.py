"""Figura: fronteira de decisao de uma arvore unica x bagging no Iris (so petalas)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

AQUI = Path(__file__).resolve().parent
CORES = ["#2a78d6", "#eb6834", "#1baf7a"]   # slots 1-3 da paleta validada
MARCAS = ["o", "s", "^"]                    # forma redundante com a cor
INK, MUTED = "#0b0b0b", "#52514e"

iris = load_iris()
X, y = iris.data[:, 2:4], iris.target       # petal length, petal width
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=42)
arvore = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
bagging = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            bootstrap=True, n_jobs=-1, random_state=42).fit(X_tr, y_tr)

xx, yy = np.meshgrid(np.linspace(0.8, 7.2, 400), np.linspace(0, 2.8, 400))
grade = np.c_[xx.ravel(), yy.ravel()]
fundo = ListedColormap(["#cfe0f5", "#fbd9cb", "#c9ecdf"])

plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#b9b8b2"})
fig, eixos = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
for ax, clf, titulo in ((eixos[0], arvore, "Árvore única"),
                        (eixos[1], bagging, "Bagging (500 árvores)")):
    Z = clf.predict(grade).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5], cmap=fundo)
    for k in range(3):
        ax.scatter(X_tr[y_tr == k, 0], X_tr[y_tr == k, 1], s=18, color=CORES[k],
                   marker=MARCAS[k], edgecolor="white", linewidth=0.5,
                   label=iris.target_names[k])
    ax.set_title(f"{titulo}\nacurácia no teste = {clf.score(X_te, y_te):.3f}".replace(".", ","),
                 fontsize=11, color=INK)
    ax.set_xlabel("comprimento da pétala (cm)")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
eixos[0].set_ylabel("largura da pétala (cm)")
eixos[1].legend(loc="upper left", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(AQUI / "figures" / "fronteiras_arvore_bagging.pdf")
print("ok")
