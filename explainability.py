import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Data prep (identical pipeline, identical random_state, to match model.py exactly) ---
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
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# --- Rebuild the SAME architecture, then load saved weights into it ---
class TitanicNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

model = TitanicNet(input_size=X_train.shape[1])
model.load_state_dict(torch.load("trained_model.pt"))
model.eval()

print("Loaded trained model successfully.")


def get_accuracy(model, X_tensor, y_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        preds = (outputs >= 0.5).float()
    return accuracy_score(y_tensor.numpy(), preds.numpy())

X_test_t = torch.tensor(X_test.values, dtype=torch.float32)

# --- Baseline accuracy, unshuffled ---
baseline_acc = get_accuracy(model, X_test_t, y_test_t)
print(f"Baseline test accuracy: {baseline_acc:.4f}\n")

# --- Permutation importance: shuffle one feature at a time ---
importances = {}
feature_names = X_test.columns.tolist()

np.random.seed(42)
for i, feature in enumerate(feature_names):
    X_test_permuted = X_test.copy()
    X_test_permuted[feature] = np.random.permutation(X_test_permuted[feature].values)

    X_permuted_t = torch.tensor(X_test_permuted.values, dtype=torch.float32)
    permuted_acc = get_accuracy(model, X_permuted_t, y_test_t)

    importance = baseline_acc - permuted_acc
    importances[feature] = importance
    print(f"{feature:<15} accuracy after shuffle: {permuted_acc:.4f}  |  importance (drop): {importance:.4f}")

# --- Plot as sorted horizontal bar chart ---
sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
names = [item[0] for item in sorted_items]
values = [item[1] for item in sorted_items]

plt.figure(figsize=(8,5))
plt.barh(names, values)
plt.xlabel("Accuracy Drop (Importance)")
plt.title("Permutation Feature Importance (Neural Network)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("plots/feature_importance.png")
plt.close()

print("\nFeature importance plot saved to plots/feature_importance.png")