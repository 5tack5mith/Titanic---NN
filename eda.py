import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/titanic.csv")

# 1. Target variable distribution
print(df["Survived"].value_counts())
print(df["Survived"].value_counts(normalize=True).round(3))

# Plot 1: survival counts
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count (0 = Died, 1 = Survived)")
plt.savefig("plots/survival_count.png")
plt.close()

# Plot 2: survival rate by sex
plt.figure(figsize=(6,4))
sns.barplot(x="Sex", y="Survived", data=df)
plt.title("Survival Rate by Sex")
plt.savefig("plots/survival_by_sex.png")
plt.close()

print("Plots saved to plots/ folder")