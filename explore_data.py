import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/titanic.csv")

print("Shape (rows, columns):", df.shape)

print("\nColumn names:", list(df.columns))

print("\nData types:\n", df.dtypes)

print("\nFirst 5 rows:\n", df.head())

# missing values per column
print("\nMissing values per column:\n", df.isnull().sum())

# missing values as a percentage of total rows
print("\nMissing values (%):\n", (df.isnull().sum() / len(df) * 100).round(2))

# --- Handling Missing Values ---

# Age: imputed with median (mean will be pulled by the outliers, preserves all rows)
df["Age"] = df["Age"].fillna(df["Age"].median())

# Embarked: imputed with mode (most frequent port), since missing count is trivial
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin: drop entirely — 77% missing makes imputation unreliable
df = df.drop(columns=["Cabin"])

# Confirm no missing values remain in the columns we're keeping
print("\nMissing values after cleaning:\n", df.isnull().sum())


# --- Feature Selection ---
# dropping PassengerId (just an index), Name and Ticket (unstructured text, not usable directly)
df = df.drop(columns=["PassengerId", "Name", "Ticket"])

# --- Encoding categorical columns ---

# Sex: binary category, label encoding is fine (no ordinal issue with only 2 classes)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Embarked: 3 categories with no real order, use one-hot to avoid implying rank
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

print("\nColumns after feature selection + encoding:\n", df.columns.tolist())
print("\nFirst 5 rows after encoding:\n", df.head())


# --- Feature Scaling ---
numeric_cols = ["Age", "Fare", "SibSp", "Parch"]

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\nFirst 5 rows after scaling:\n", df.head())