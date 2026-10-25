"""Figura: fronteira da arvore (prof. 3) com fronteira verdadeira diagonal, antes e depois de rodar 45 graus."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

AQUI = Path(__file__).resolve().parent
CORES, MARCAS = ["#2a78d6", "#eb6834"], ["o", "s"]
INK, MUTED = "#0b0b0b", "#52514e"
th = np.pi / 4
R45 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
rng = np.random.default_rng(1)
X = rng.normal(size=(400, 2))
y = (X[:, 0] + X[:, 1] + rng.normal(scale=0.1, size=400) > 0).astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)

xx, yy = np.meshgrid(np.linspace(-3, 3, 400), np.linspace(-3, 3, 400))
grade = np.c_[xx.ravel(), yy.ravel()]
fundo = ListedColormap(["#cfe0f5", "#fbd9cb"])
plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#b9b8b2"})
fig, eixos = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, rodar, titulo in ((eixos[0], False, "eixos originais"),
                          (eixos[1], True, "árvore treinada com dados rodados 45°")):
    A, T, G = (X_tr @ R45.T, X_te @ R45.T, grade @ R45.T) if rodar else (X_tr, X_te, grade)
    clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(A, y_tr)
    Z = clf.predict(G).reshape(xx.shape)           # em cada ponto da grade ORIGINAL
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], cmap=fundo)
    for k in (0, 1):
        ax.scatter(X_tr[y_tr == k, 0], X_tr[y_tr == k, 1], s=12, color=CORES[k],
                   marker=MARCAS[k], edgecolor="white", linewidth=0.4)
    ax.plot([-3, 3], [3, -3], color=INK, lw=1, ls="--")
    ax.set_title(f"{titulo}\nacurácia no teste = {clf.score(T, y_te):.3f}".replace(".", ","),
                 fontsize=10.5, color=INK)
    ax.set_xlabel("$x_1$")
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
eixos[0].set_ylabel("$x_2$")
fig.tight_layout()
fig.savefig(AQUI / "figures" / "rotacao.pdf")
print("ok")
