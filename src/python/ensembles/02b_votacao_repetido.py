import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

X, y = load_iris(return_X_y=True)
membros = [("lr",  LogisticRegression(max_iter=200)),
           ("arv", DecisionTreeClassifier(random_state=0)),
           ("knn", KNeighborsClassifier())]
modelos = dict(membros, hard=VotingClassifier(membros, voting="hard"),
                        soft=VotingClassifier(membros, voting="soft"))

acc = {nome: [] for nome in modelos}
for semente in range(100):                # 100 divisoes treino/teste
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=semente)
    for nome, clf in modelos.items():
        acc[nome].append(clf.fit(X_tr, y_tr).score(X_te, y_te))

for nome, a in acc.items():
    print(f"{nome:5s} media = {np.mean(a):.3f}   dp = {np.std(a):.3f}")
