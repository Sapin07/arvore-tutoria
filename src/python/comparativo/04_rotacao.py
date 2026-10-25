"""Rotacao dos eixos. (a) Dados sinteticos com fronteira diagonal x1 + x2 = 0;
(b) Iris com os 4 eixos rotacionados por uma matriz ortogonal aleatoria."""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

th = np.pi / 4
R45 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

def media_dp(v):
    return f"{np.mean(v):.3f} ± {np.std(v):.3f}"

print("(a) sinteticos: y = 1 se x1 + x2 + ruido > 0   (50 divisoes)")
arv, arv_rot, logit = [], [], []
for s in range(50):
    rng = np.random.default_rng(s)
    X = rng.normal(size=(400, 2))
    y = (X[:, 0] + X[:, 1] + rng.normal(scale=0.1, size=400) > 0).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=s)
    arv.append(DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_tr, y_tr).score(X_te, y_te))
    arv_rot.append(DecisionTreeClassifier(max_depth=3, random_state=0)
                   .fit(X_tr @ R45.T, y_tr).score(X_te @ R45.T, y_te))
    logit.append(LogisticRegression(C=1e4).fit(X_tr, y_tr).score(X_te, y_te))
print("  arvore (prof. 3), eixos originais :", media_dp(arv))
print("  arvore (prof. 3), eixos rodados 45:", media_dp(arv_rot))
print("  regressao logistica               :", media_dp(logit))

print("(b) Iris, 4 eixos rodados por matriz ortogonal aleatoria (50 divisoes)")
Xi, yi = load_iris(return_X_y=True)
Z = StandardScaler().fit_transform(Xi)
Q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(4, 4)))
a, b = [], []
for s in range(50):
    idx_tr, idx_te = train_test_split(np.arange(150), test_size=0.25, stratify=yi, random_state=s)
    for lista, M in ((a, Z), (b, Z @ Q)):
        lista.append(DecisionTreeClassifier(max_depth=3, random_state=0)
                     .fit(M[idx_tr], yi[idx_tr]).score(M[idx_te], yi[idx_te]))
print("  arvore, eixos originais:", media_dp(a))
print("  arvore, eixos rodados  :", media_dp(b))
