from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=42)

arvore = DecisionTreeClassifier(random_state=42)
bagging = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            bootstrap=True, n_jobs=-1, random_state=42)

for nome, clf in (("arvore", arvore), ("bagging", bagging)):
    clf.fit(X_tr, y_tr)
    print(f"{nome:8s} treino = {clf.score(X_tr, y_tr):.3f}   "
          f"teste = {clf.score(X_te, y_te):.3f}")
