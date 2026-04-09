import pandas as pd
from sklearn.model_selection import train_test_split
from skrebate import ReliefF
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 1. Cargar dataset
df = pd.read_excel("C:/Users/emman/Downloads/dataset_project.xlsx")

# 2. Mostrar primeras filas
print("Primeras filas del dataset:")
print(df.head())

# 3. Preparar variables
# Variables de entrada
X = df.drop(["user_id", "activity_level"], axis=1).astype(float)

# Variable objetivo
y = df["activity_level"]

# 4. Convertir variable objetivo a números
y = y.map({
    "low": 0,
    "medium": 1,
    "high": 2
})

# 5. Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTamaño de entrenamiento:", X_train.shape)
print("Tamaño de prueba:", X_test.shape)

# 6. Aplicar Relief
relief = ReliefF(n_neighbors=10)

relief.fit(X_train.values, y_train.values)

# 7. Obtener importancia de características
feature_importance = relief.feature_importances_

print("\nImportancia de características según Relief:")
for feature, importance in zip(X.columns, feature_importance):
    print(f"{feature}: {importance}")

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy del modelo:", accuracy)

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(16, 8))

annotations = plot_tree(
    model,
    feature_names=X.columns,
    class_names=["low", "medium", "high"],
    filled=True,
    rounded=True,
    impurity=False,   # quita gini
    fontsize=12
)

# Limpiar texto de cada nodo para quitar "samples" y "value"
for ann in annotations:
    text = ann.get_text().split("\n")

    # Si es nodo interno: dejar condición + clase
    if len(text) > 1 and "<=" in text[0]:
        new_text = text[0] + "\n" + text[-1]
    else:
        # Si es hoja: dejar solo la clase
        new_text = text[-1]

    ann.set_text(new_text)

plt.title("Árbol de Decisión", fontsize=16)
plt.tight_layout()
plt.show()
