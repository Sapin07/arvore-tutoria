"""Figura: arvore de regressao (IMC -> progressao da doenca) com profundidades 2, 5 e ilimitada."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

AQUI = Path(__file__).resolve().parent
AZUL, LARANJA = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"
dados = load_diabetes(as_frame=True).frame
X, y = dados[["bmi"]].values, dados["target"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
grade = np.linspace(X.min(), X.max(), 600).reshape(-1, 1)

plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#b9b8b2"})
fig, eixos = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
for ax, h, titulo in zip(eixos, (2, 5, None), ("profundidade 2", "profundidade 5", "sem limite")):
    m = DecisionTreeRegressor(max_depth=h, random_state=42).fit(X_tr, y_tr)
    rmse = np.sqrt(np.mean((m.predict(X_te) - y_te) ** 2))
    ax.scatter(X_tr, y_tr, s=10, color=AZUL, alpha=0.45, linewidth=0)
    ax.step(grade, m.predict(grade), where="mid", color=LARANJA, lw=2)
    ax.set_title(f"{titulo}: {m.get_n_leaves()} folhas\nRMSE teste = {rmse:.1f}".replace(".", ","),
                 fontsize=10.5, color=INK)
    ax.set_xlabel("IMC (padronizado)")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
eixos[0].set_ylabel("progressão da doença")
fig.tight_layout()
fig.savefig(AQUI / "figures" / "degraus.pdf")
print("ok")
