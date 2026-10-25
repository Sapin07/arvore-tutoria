"""Figura: acuracia de teste do bagging no Iris (4 variaveis) em funcao de B (30 divisoes)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

AQUI = Path(__file__).resolve().parent
AZUL, LARANJA = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"
X, y = load_iris(return_X_y=True)
Bs = [1, 2, 3, 5, 10, 20, 50, 100, 200]
reps = range(30)

arv = np.zeros(len(reps))
bag = np.zeros((len(reps), len(Bs)))
for r in reps:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=r)
    arv[r] = DecisionTreeClassifier(random_state=r).fit(X_tr, y_tr).score(X_te, y_te)
    for j, B in enumerate(Bs):
        m = BaggingClassifier(DecisionTreeClassifier(), n_estimators=B,
                              n_jobs=-1, random_state=r).fit(X_tr, y_tr)
        bag[r, j] = m.score(X_te, y_te)

mu, dp = bag.mean(axis=0), bag.std(axis=0)
with open(AQUI / "outputs" / "fig_acuracia_vs_B.txt", "w") as f:
    f.write(f"arvore unica: media={arv.mean():.3f} dp={arv.std():.3f}\n")
    for B, m, s in zip(Bs, mu, dp):
        f.write(f"B={B:>3}: media={m:.3f} dp={s:.3f}\n")

plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#b9b8b2"})
fig, ax = plt.subplots(figsize=(6.6, 3.5))
ax.axhline(arv.mean(), color=LARANJA, lw=2, label="árvore única (média das 30 divisões)")
ax.fill_between(Bs, mu - dp, mu + dp, color=AZUL, alpha=0.15, linewidth=0)
ax.plot(Bs, mu, color=AZUL, lw=2, marker="o", ms=5, markeredgecolor="white",
        label="bagging (média ± 1 dp)")
ax.set_xscale("log")
ax.set_xlabel("número de árvores $B$")
ax.set_ylabel("acurácia no teste")
ax.legend(loc="lower right", frameon=False, fontsize=10)
for lado in ("top", "right"):
    ax.spines[lado].set_visible(False)
ax.grid(axis="y", color="#e5e4de", lw=0.8)
fig.tight_layout()
fig.savefig(AQUI / "figures" / "acuracia_vs_B.pdf")
print("ok")
