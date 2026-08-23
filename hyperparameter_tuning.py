import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score

# --- Data prep (same pipeline as model.py) ---
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

X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

class FlexibleNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation="relu", dropout=0.0):
        super().__init__()
        layers = []
        prev_size = input_size

        act_fn = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "leaky_relu": nn.LeakyReLU()}[activation]

        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = h

        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_and_evaluate(model, learning_rate=0.001, epochs=200):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        test_preds = (test_outputs >= 0.5).float()

    y_true = y_test_t.numpy()
    y_pred = test_preds.numpy()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

# --- Grid search ---
learning_rates = [0.001, 0.01, 0.0001]
architectures = {
    "small [8]": [8],
    "medium [16,8]": [16, 8],
    "large [32,16]": [32, 16],
}

grid_results = []

for lr in learning_rates:
    for arch_name, hidden_sizes in architectures.items():
        torch.manual_seed(42)
        model = FlexibleNet(input_size=8, hidden_sizes=hidden_sizes, activation="relu", dropout=0.0)
        metrics = train_and_evaluate(model, learning_rate=lr)
        grid_results.append({
            "learning_rate": lr,
            "architecture": arch_name,
            **metrics
        })

# --- Print as a sorted table, best F1 first ---
grid_results_sorted = sorted(grid_results, key=lambda r: r["f1"], reverse=True)

print(f"{'LR':<10}{'Architecture':<16}{'Accuracy':<12}{'Recall':<12}{'F1':<12}")
print("-" * 62)
for r in grid_results_sorted:
    print(f"{r['learning_rate']:<10}{r['architecture']:<16}{r['accuracy']:<12.4f}{r['recall']:<12.4f}{r['f1']:<12.4f}")

print(f"\nBest combination: LR={grid_results_sorted[0]['learning_rate']}, Architecture={grid_results_sorted[0]['architecture']}")