import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Load and preprocess (repeating Part 1 pipeline) ---
df = pd.read_csv("data/titanic.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["Cabin", "PassengerId", "Name", "Ticket"])

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# convert one-hot True/False to actual 0/1 ints for PyTorch
df["Embarked_Q"] = df["Embarked_Q"].astype(int)
df["Embarked_S"] = df["Embarked_S"].astype(int)

# --- Split features (X) and target (y) ---
X = df.drop(columns=["Survived"])
y = df["Survived"]

# --- Train / Validation / Test split: 70/15/15 ---
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# --- Scale AFTER splitting, fit only on training data ---
scaler = StandardScaler()
numeric_cols = ["Age", "Fare", "SibSp", "Parch"]

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# --- Convert everything to PyTorch tensors ---
X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
X_test_t = torch.tensor(X_test.values, dtype=torch.float32)

y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

print("Train:", X_train_t.shape, y_train_t.shape)
print("Val:  ", X_val_t.shape, y_val_t.shape)
print("Test: ", X_test_t.shape, y_test_t.shape)


import torch.nn as nn

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

model = TitanicNet(input_size=X_train_t.shape[1])
print(model)

# --- Loss function and optimizer ---
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 300

train_losses = []
val_losses = []
train_accs = []
val_accs = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_loss = criterion(val_outputs, y_val_t)

        train_preds = (outputs >= 0.5).float()
        val_preds = (val_outputs >= 0.5).float()
        train_acc = (train_preds == y_train_t).float().mean().item()
        val_acc = (val_preds == y_val_t).float().mean().item()

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig("plots/loss_curve.png")
plt.close()

torch.save(model.state_dict(), "trained_model.pt")
print("Model saved to trained_model.pt")

print("Loss curve saved to plots/loss_curve.png")

plt.figure(figsize=(8,5))
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.savefig("plots/accuracy_curve.png")
plt.close()

print("Accuracy curve saved to plots/accuracy_curve.png")

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t)
    test_preds = (test_outputs >= 0.5).float()

y_true = y_test_t.numpy()
y_pred = test_preds.numpy()

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print(f"\nTest Accuracy:  {acc:.4f}")
print(f"Test Precision: {prec:.4f}")
print(f"Test Recall:    {rec:.4f}")
print(f"Test F1-score:  {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)

# --- Part 8: Predictions on unseen samples ---
new_passengers = pd.DataFrame({
    "Pclass": [1, 3, 2, 3, 1],
    "Sex": [1, 0, 1, 0, 0],
    "Age": [28, 22, 4, 35, 60],
    "SibSp": [0, 0, 1, 0, 1],
    "Parch": [0, 0, 1, 0, 0],
    "Fare": [100, 7.5, 30, 8, 90],
    "Embarked_Q": [0, 0, 0, 1, 0],
    "Embarked_S": [1, 1, 1, 0, 1],
})

new_passengers[numeric_cols] = scaler.transform(new_passengers[numeric_cols])
new_passengers_t = torch.tensor(new_passengers.values, dtype=torch.float32)

model.eval()
with torch.no_grad():
    new_probs = model(new_passengers_t)
    new_preds = (new_probs >= 0.5).float()

for i in range(len(new_passengers)):
    print(f"Passenger {i+1}: probability={new_probs[i].item():.4f}, prediction={'Survived' if new_preds[i].item()==1 else 'Did Not Survive'}")