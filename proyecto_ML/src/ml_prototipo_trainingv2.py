
import pandas as pd
from sklearn.model_selection import train_test_split
from skrebate import ReliefF
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


#df = pd.read_excel("C:/Users/emman/Downloads/database_v1.1.xlsx")
df = pd.read_excel("C:/Users/emman/Downloads/database_v2.0.xlsx")

print("Primeras filas del dataset:")
print(df.head())

#X = df.drop(["user_id", "activity_level"], axis=1).astype(float)
X = df.drop(["user_id", "activity_level", "activity_numeric" ], axis=1).astype(float)

y = df["activity_level"].map({
    "low": 0,
    "medium": 1,
    "high": 2
})


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTamaño de entrenamiento:", X_train.shape)
print("Tamaño de prueba:", X_test.shape)


relief = ReliefF(n_neighbors=10)
relief.fit(X_train.values, y_train.values)

feature_importance = relief.feature_importances_

print("\nImportancia de características según Relief:")
for feature, importance in zip(X.columns, feature_importance):
    print(f"{feature}: {importance}")

model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy del modelo:", accuracy)

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))


plt.figure(figsize=(20, 10))

annotations = plot_tree(
    model,
    feature_names=X.columns,
    class_names=["low", "medium", "high"],
    filled=True,
    rounded=True,
    impurity=False,
    fontsize=11
)

for ann in annotations:
    text = ann.get_text().split("\n")

    if len(text) > 1 and "<=" in text[0]:
        new_text = text[0] + "\n" + text[-1]
    else:
        new_text = text[-1]

    ann.set_text(new_text)

plt.title("Árbol de Decisión", fontsize=16)
plt.tight_layout()
plt.savefig("decision_tree_clean.png", dpi=300, bbox_inches="tight")
plt.show()
