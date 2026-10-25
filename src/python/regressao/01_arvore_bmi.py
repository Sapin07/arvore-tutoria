import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text

dados = load_diabetes(as_frame=True).frame        # 442 pacientes, 10 variaveis
X, y = dados[["bmi"]], dados["target"]            # so o IMC, para poder desenhar
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

arvore = DecisionTreeRegressor(max_depth=2, random_state=42).fit(X_tr, y_tr)
print(export_text(arvore, feature_names=["bmi"]))
rmse = np.sqrt(np.mean((arvore.predict(X_te) - y_te) ** 2))
print(f"RMSE no teste = {rmse:.1f}   (desvio-padrao de y = {y_te.std():.1f})")
