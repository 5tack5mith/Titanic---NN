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

## Part 3+ (to be filled in as we go)