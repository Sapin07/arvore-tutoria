from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=42)

ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),   # stumps
                         n_estimators=200, learning_rate=1.0, random_state=42)
ada.fit(X_tr, y_tr)

for B, acc in enumerate(ada.staged_score(X_te, y_te), start=1):
    if B in (1, 10, 50, 200):
        print(f"B = {B:>3}: acuracia no teste = {acc:.3f}")
