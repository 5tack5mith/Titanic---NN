import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

# --- Data prep (same pipeline as model.py / experiments.py) ---
df = pd.read_csv("data/titanic.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["Cabin", "PassengerId", "Name", "Ticket"])

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
df["Embarked_Q"] = df["Embarked_Q"].astype(int)
df["Embarked_S"] = df["Embarked_S"].astype(int)

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

scaler = StandardScaler()
numeric_cols = ["Age", "Fare", "SibSp", "Parch"]
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# --- Classical ML models ---
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5),
}

results = []

for name, clf in models.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    })

# Add NN baseline result for direct comparison (from Part 5 evaluation)
results.append({"model": "Neural Network (baseline)", "accuracy": 0.7463, "recall": 0.6863, "f1": 0.6731})

# --- Print comparison table ---
print(f"{'Model':<28}{'Accuracy':<12}{'Recall':<12}{'F1':<12}")
print("-" * 64)
for r in results:
    print(f"{r['model']:<28}{r['accuracy']:<12.4f}{r['recall']:<12.4f}{r['f1']:<12.4f}")