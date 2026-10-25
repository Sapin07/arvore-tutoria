from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=42)

membros = [("lr",  LogisticRegression(max_iter=200)),
           ("arv", DecisionTreeClassifier(random_state=42)),
           ("knn", KNeighborsClassifier())]
hard = VotingClassifier(membros, voting="hard")
soft = VotingClassifier(membros, voting="soft")

for nome, clf in membros + [("hard", hard), ("soft", soft)]:
    clf.fit(X_tr, y_tr)
    print(f"{nome:5s} acuracia no teste = {clf.score(X_te, y_te):.3f}")
