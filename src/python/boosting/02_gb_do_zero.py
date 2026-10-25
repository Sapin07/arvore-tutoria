import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
rmse = lambda pred: np.sqrt(np.mean((pred - y_te) ** 2))
B, lam, d = 100, 0.1, 2

# Algoritmo 9 "do zero": cada arvore ajusta os residuos atuais
f_tr = np.full(len(y_tr), y_tr.mean())            # f0 = media de y
f_te = np.full(len(y_te), y_tr.mean())
for b in range(B):
    r = y_tr - f_tr                               # residuos
    arvore = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X_tr, r)
    f_tr += lam * arvore.predict(X_tr)
    f_te += lam * arvore.predict(X_te)
print(f"do zero  : RMSE no teste = {rmse(f_te):.1f}")

# a mesma coisa com a biblioteca
gb = GradientBoostingRegressor(n_estimators=B, learning_rate=lam,
                               max_depth=d, random_state=0).fit(X_tr, y_tr)
print(f"sklearn  : RMSE no teste = {rmse(gb.predict(X_te)):.1f}")
