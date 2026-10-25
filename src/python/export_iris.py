"""Exporta o Iris do scikit-learn para data/iris.csv (mesmo arquivo para Python e R)."""
from pathlib import Path
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame.rename(columns={
    "sepal length (cm)": "sepal_length", "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length", "petal width (cm)": "petal_width"})
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
destino = Path(__file__).resolve().parents[2] / "data" / "iris.csv"
df.drop(columns="target").to_csv(destino, index=False)
print("escrito:", destino, df.shape)
