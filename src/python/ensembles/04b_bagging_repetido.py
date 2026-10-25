import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

X, y = load_iris(return_X_y=True)
problemas = {"4 variaveis": [0, 1, 2, 3], "so sepalas": [0, 1]}

for nome, cols in problemas.items():
    acc = {"arvore": [], "bagging": []}
    for semente in range(100):            # 100 divisoes treino/teste
        X_tr, X_te, y_tr, y_te = train_test_split(
            X[:, cols], y, test_size=0.25, stratify=y, random_state=semente)
        arv = DecisionTreeClassifier(random_state=semente)
        bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=200,
                                random_state=semente)
        acc["arvore"].append(arv.fit(X_tr, y_tr).score(X_te, y_te))
        acc["bagging"].append(bag.fit(X_tr, y_tr).score(X_te, y_te))
    for modelo, a in acc.items():
        print(f"{nome:12s} {modelo:8s} media = {np.mean(a):.3f}   dp = {np.std(a):.3f}")
