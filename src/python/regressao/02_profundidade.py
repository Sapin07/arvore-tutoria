import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

dados = load_diabetes(as_frame=True).frame
X, y = dados[["bmi"]].values, dados["target"].values

def rmse(m, X, y):
    return np.sqrt(np.mean((m.predict(X) - y) ** 2))

print("max_depth  folhas   RMSE treino   RMSE teste   (media de 30 divisoes)")
for h in (1, 2, 3, 5, 8, None):
    tr, te, nf = [], [], []
    for semente in range(30):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                                  random_state=semente)
        m = DecisionTreeRegressor(max_depth=h, random_state=0).fit(X_tr, y_tr)
        tr.append(rmse(m, X_tr, y_tr)); te.append(rmse(m, X_te, y_te))
        nf.append(m.get_n_leaves())
    print(f"{str(h):>9s}  {np.mean(nf):6.0f}   {np.mean(tr):11.1f}   {np.mean(te):10.1f}")
