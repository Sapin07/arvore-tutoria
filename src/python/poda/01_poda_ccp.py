from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=42)

# sequencia de alphas efetivos da arvore completa (so no treino)
caminho = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X_tr, y_tr)
alphas = caminho.ccp_alphas[:-1]            # o ultimo alpha deixa so a raiz

# alpha escolhido por validacao cruzada de 5 partes
busca = GridSearchCV(DecisionTreeClassifier(random_state=0),
                     {"ccp_alpha": alphas}, cv=5).fit(X_tr, y_tr)

completa = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
podada = busca.best_estimator_
print(f"{len(alphas)} valores de alpha; escolhido: {busca.best_params_['ccp_alpha']:.4f}")
for nome, m in (("completa", completa), ("podada", podada)):
    print(f"{nome:8s} folhas = {m.get_n_leaves():2d}   teste = {m.score(X_te, y_te):.3f}")
