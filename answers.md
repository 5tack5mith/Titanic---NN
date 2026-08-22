# Titanic Neural Network — Task Notes & Answers

## Part 1: Preprocessing

**Missing values:**
- Age (19.87% missing) → imputed with median. Median chosen over mean because Age
  is right-skewed with outliers; mean would be pulled upward by older passengers,
  median is robust to this.
- Embarked (0.22% missing) → imputed with mode (most frequent port). Missing count
  is trivial (2 rows), any reasonable strategy works.
- Cabin (77.10% missing) → dropped entirely. At this level of missingness,
  imputation would be mostly fabricated data rather than genuine signal.

**Feature selection:**
- Dropped: PassengerId (arbitrary index, no signal), Name (raw text, not directly
  usable), Ticket (unstructured alphanumeric codes, mostly unique per passenger).
- Kept: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked — each has a defensible
  link to survival (socioeconomic status, "women/children first" policy, family
  group evacuation behavior).

**Encoding:**
- Sex → label encoded (male=0, female=1). Fine for binary categories, no ordinal
  issue since there's no natural order between two classes.
- Embarked → one-hot encoded (drop_first=True). Necessary since it has 3 unordered
  categories (S, C, Q) — label encoding would falsely imply order/distance between
  ports that doesn't exist.

**Feature scaling:**
- StandardScaler applied to Age, Fare, SibSp, Parch (continuous numeric columns).
- Why it matters for NNs: features on very different raw scales (Age ~0-80 vs
  Fare ~0-512) cause the network to implicitly treat larger-magnitude features as
  more important regardless of actual relevance, and can destabilize gradient
  descent. Scaling puts all features on a comparable footing (mean 0, std 1).

## Part 2: EDA

**Target variable distribution:** 549 died (61.6%), 342 survived (38.4%) —
moderately imbalanced. This sets a naive baseline: a model that always predicts
"died" would score 61.6% accuracy with zero real learning. Any trained model
needs to clearly beat this to prove it learned something.

**Observation 1:** Class imbalance (61.6% / 38.4%) means accuracy alone can be
misleading — see baseline reasoning above.

**Observation 2:** Sex is a dominant predictor — female survival rate ~74% vs
male survival rate ~19%, a ~4x difference with non-overlapping confidence
intervals. Consistent with the historical "women and children first" evacuation
policy. Strong early signal the model should heavily weight this feature.

## Part 3-4: Model Architecture & Training

**Architecture:** 3-layer feedforward network (MLP) — input(8) -> 16 -> 8 -> output(1),
ReLU activations on hidden layers, Sigmoid on output (binary classification probability).
~289 learnable parameters total. Small architecture deliberately chosen given the small
dataset (623 training rows) to avoid excessive overfitting risk.

**Loss function:** Binary Cross Entropy (BCELoss) — standard for binary classification,
penalizes confident wrong predictions more heavily than uncertain ones.

**Optimizer:** Adam, learning rate 0.001 — adaptive optimizer, generally faster and more
stable convergence than plain SGD, 0.001 is a standard safe starting learning rate for Adam.

**Epochs:** Extended from 100 to 300 after observing loss was still steadily decreasing
at 100 epochs (hadn't converged yet).

**Training/Validation loss behavior (300 epochs):**
- Both losses decreased steadily and smoothly through ~epoch 120, then the curves
  crossed — validation loss became consistently lower than training loss for the
  remainder of training, ending at Train 0.404 / Val 0.359.
- Both curves flattened/plateaued by roughly epoch 250-300, suggesting the model had
  largely converged at this architecture/epoch combination.
- No overfitting observed: overfitting specifically requires validation loss to rise
  while training loss falls, which did not happen — both trended down together.
- Val loss being lower than train loss is likely explained by the specific random
  train/val split (134 validation rows is a small sample) happening to contain a
  favorable/easier mix of cases, rather than any methodological issue.