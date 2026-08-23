# Titanic Survival Prediction — Neural Network Classifier

A binary classification project predicting Titanic passenger survival, built as
an AI/ML recruitment task. Implements a feedforward neural network (PyTorch)
from scratch, with full preprocessing, EDA, evaluation, experimentation, and
three bonus extensions (hyperparameter tuning, classical ML comparison, and
model explainability).

Full reasoning, answers to all task questions, and detailed analysis are in
[`answers.md`](./answers.md).

## Results Summary

**Baseline Neural Network (test set):**
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 74.63% |
| Recall    | 68.63% |
| F1-score  | 0.6731 |

Naive baseline (always predict majority class): 61.6% accuracy — the model
clearly outperforms this, confirming genuine learned signal.

**Strongest predictor (via permutation importance):** `Sex` — shuffling this
single feature drops model accuracy by ~19.4 percentage points, roughly 4-5x
more impactful than any other feature.

**Model comparison:** Logistic Regression matched the neural network's recall
exactly (68.63%), suggesting survival patterns in this dataset are largely
linear/simple — full discussion in `answers.md`, Bonus 2.

## Project Structure
```
titanic-nn-project/
├── data/                        # dataset folder (auto-created by download_data.py)
├── plots/                       # all generated charts (EDA, loss/accuracy curves, feature importance)
├── venv/                        # virtual environment (not committed, see .gitignore)
├── sanity_check.py              # initial PyTorch install verification
├── download_data.py             # fetches and saves the Titanic dataset
├── explore_data.py              # Part 1: preprocessing exploration
├── eda.py                       # Part 2: exploratory data analysis
├── model.py                     # Parts 3-5, 6, 8: model build, train, evaluate, curves, predict
├── experiments.py               # Part 7: 5-way architecture/hyperparameter experiment
├── hyperparameter_tuning.py     # Bonus 1: grid search over LR x architecture
├── classical_comparison.py      # Bonus 2: NN vs Logistic Regression / Decision Tree / Random Forest
├── explainability.py            # Bonus 3: permutation feature importance
├── trained_model.pt             # saved weights of the trained baseline model
├── answers.md                   # full write-up, reasoning, and task question answers
├── README.md
└── requirements.txt
```
## How to Run

```bash
# 1. Set up environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

pip install -r requirements.txt

# 2. Download the dataset
python download_data.py

# 3. Run the main pipeline (preprocessing -> training -> evaluation -> predictions)
python model.py

# 4. Optional: EDA, experiments, and bonus tasks
python eda.py
python experiments.py
python hyperparameter_tuning.py
python classical_comparison.py
python explainability.py
```

## Approach

1. **Preprocessing:** handled missing values (median imputation for `Age`, mode
   for `Embarked`, dropped `Cabin` at 77% missing), dropped non-predictive
   columns (`PassengerId`, `Name`, `Ticket`), encoded categoricals (label
   encoding for binary `Sex`, one-hot for unordered `Embarked`), and scaled
   numeric features with `StandardScaler` — fit only on training data to avoid
   leakage.
2. **EDA:** confirmed class imbalance (61.6%/38.4%) and a strong survival gap
   by sex (~74% female vs ~19% male).
3. **Model:** a 3-layer feedforward network (8 → 16 → 8 → 1), ReLU hidden
   activations, sigmoid output, trained with Adam and Binary Cross Entropy loss
   over 300 epochs, using a 70/15/15 train/val/test split.
4. **Evaluation:** accuracy, precision, recall, F1, and confusion matrix on a
   held-out test set, with explicit reasoning on why accuracy alone is
   insufficient given class imbalance.
5. **Experimentation:** compared 5 architecture/hyperparameter variations
   against baseline, selecting a final model based on recall (prioritized
   because false negatives were judged more costly), not just accuracy.
6. **Bonus:** systematic hyperparameter grid search, comparison against three
   classical ML models, and permutation-based feature importance analysis.

Full reasoning behind every decision — including alternatives considered — is
documented in [`answers.md`](./answers.md).