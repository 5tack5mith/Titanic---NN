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

def train_and_evaluate(model, learning_rate=0.001, epochs=300):
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

results = {}

# Baseline (matches model.py: 2 hidden layers [16,8], ReLU, no dropout, lr=0.001)
torch.manual_seed(42)
baseline_model = FlexibleNet(input_size=8, hidden_sizes=[16, 8], activation="relu", dropout=0.0)
results["Baseline (2 layers, ReLU)"] = train_and_evaluate(baseline_model, learning_rate=0.001)

# Experiment A: 1 hidden layer instead of 2
torch.manual_seed(42)
exp_a = FlexibleNet(input_size=8, hidden_sizes=[16], activation="relu", dropout=0.0)
results["Exp A: 1 hidden layer"] = train_and_evaluate(exp_a, learning_rate=0.001)

# Experiment B: Tanh instead of ReLU
torch.manual_seed(42)
exp_b = FlexibleNet(input_size=8, hidden_sizes=[16, 8], activation="tanh", dropout=0.0)
results["Exp B: Tanh activation"] = train_and_evaluate(exp_b, learning_rate=0.001)

# Experiment C: higher learning rate
torch.manual_seed(42)
exp_c = FlexibleNet(input_size=8, hidden_sizes=[16, 8], activation="relu", dropout=0.0)
results["Exp C: lr=0.01"] = train_and_evaluate(exp_c, learning_rate=0.01)

# Experiment D: add Dropout
torch.manual_seed(42)
exp_d = FlexibleNet(input_size=8, hidden_sizes=[16, 8], activation="relu", dropout=0.3)
results["Exp D: Dropout 0.3"] = train_and_evaluate(exp_d, learning_rate=0.001)

# Experiment E: more neurons per layer
torch.manual_seed(42)
exp_e = FlexibleNet(input_size=8, hidden_sizes=[32, 16], activation="relu", dropout=0.0)
results["Exp E: 32-16 neurons"] = train_and_evaluate(exp_e, learning_rate=0.001)

# --- Print comparison table ---
print(f"{'Model':<28}{'Accuracy':<12}{'Recall':<12}{'F1':<12}")
print("-" * 64)
for name, metrics in results.items():
    print(f"{name:<28}{metrics['accuracy']:<12.4f}{metrics['recall']:<12.4f}{metrics['f1']:<12.4f}")