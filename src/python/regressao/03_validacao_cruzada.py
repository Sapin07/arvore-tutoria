from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor

dados = load_diabetes(as_frame=True).frame
X, y = dados[["bmi"]], dados["target"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

grade = {"max_depth": [1, 2, 3, 4, 5, 6, 8, None],
         "min_samples_leaf": [1, 5, 10, 20, 40]}
busca = GridSearchCV(DecisionTreeRegressor(random_state=0), grade, cv=5,
                     scoring="neg_root_mean_squared_error").fit(X_tr, y_tr)

print("melhores hiperparametros:", busca.best_params_)
print(f"RMSE (validacao cruzada) = {-busca.best_score_:.1f}")
print(f"RMSE no teste            = {-busca.score(X_te, y_te):.1f}")
