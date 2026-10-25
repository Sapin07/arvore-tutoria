"""Diabetes (442 pacientes, 10 variaveis): arvores x baselines em regressao, 50 divisoes."""
from pathlib import Path
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

X, y = load_diabetes(return_X_y=True)           # X ja vem padronizado
modelos = {
    "Regressão linear":      lambda: LinearRegression(),
    "KNN ($k=15$)":          lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(15)),
    "SVR (RBF, $C=100$)":    lambda: make_pipeline(StandardScaler(), SVR(C=100)),
    "Árvore (prof.\\ 3)":     lambda: DecisionTreeRegressor(max_depth=3, random_state=0),
    "Árvore (completa)":     lambda: DecisionTreeRegressor(random_state=0),
    "Floresta ($B=300$)":    lambda: RandomForestRegressor(300, min_samples_leaf=5, random_state=0, n_jobs=-1),
    "Gradient boosting":     lambda: GradientBoostingRegressor(n_estimators=100, learning_rate=0.05,
                                                               max_depth=2, random_state=0),
}
res = {n: {"rmse": [], "r2": []} for n in modelos}
for semente in range(50):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=semente)
    for nome, fabrica in modelos.items():
        pred = fabrica().fit(X_tr, y_tr).predict(X_te)
        res[nome]["rmse"].append(np.sqrt(np.mean((pred - y_te) ** 2)))
        res[nome]["r2"].append(1 - np.sum((y_te - pred) ** 2) / np.sum((y_te - y_te.mean()) ** 2))

virg1 = lambda v: f"{v:.1f}".replace(".", ",")
virg2 = lambda v: f"{v:.2f}".replace(".", ",")
linhas = []
for nome, r in res.items():
    print(f"{nome:24s} RMSE = {np.mean(r['rmse']):.1f} ± {np.std(r['rmse']):.1f}   "
          f"R2 = {np.mean(r['r2']):.2f} ± {np.std(r['r2']):.2f}")
    linhas.append(f"{nome} & {virg1(np.mean(r['rmse']))} $\\pm$ {virg1(np.std(r['rmse']))}"
                  f" & {virg2(np.mean(r['r2']))} $\\pm$ {virg2(np.std(r['r2']))} \\\\")
tex = ("\\begin{tabular}{@{}lcc@{}}\n\\toprule\n"
       "Método & RMSE & $R^2$ \\\\\n\\midrule\n" + "\n".join(linhas) +
       "\n\\bottomrule\n\\end{tabular}\n")
(Path(__file__).resolve().parent / "outputs" / "tabela_baselines_regressao.tex").write_text(tex, encoding="utf-8")
