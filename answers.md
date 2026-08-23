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

## Part 5: Model Evaluation

**Test set results:**
- Accuracy:  76.12%
- Precision: 69.39%
- Recall:    66.67%
- F1-score:  68.00%

**Confusion Matrix:**

[[68 15]
[17 34]]

(TN=68, FP=15, FN=17, TP=34)

**Interpretation:** 76.12% accuracy is meaningfully above the 61.6% naive baseline
(always predicting "died"), confirming the model learned real patterns rather than
just exploiting class imbalance. FN (17) slightly exceeds FP (15), meaning the model
is marginally more prone to missing real survivors than falsely flagging non-survivors.

**Question 1: Is accuracy alone sufficient to judge this model?**
No. 76.12% accuracy looks solid in isolation, but it hides that the model missed
17 of 51 actual survivors (a 33% miss rate on the positive class) — a weakness
invisible from the accuracy number alone. Given the moderate class imbalance
(61.6%/38.4%), accuracy can mask uneven performance across classes.

**Question 2: What does the confusion matrix tell you?**
It breaks down performance by error type, not just error count. Here it shows the
model makes both false positive (15) and false negative (17) errors at similar
rates, with a slight skew toward missing real survivors (FN > FP). This is more
diagnostic than accuracy alone, since it shows *how* the model is wrong, not just
how often.

**Question 3: Which metric would you prioritize if missing a positive prediction
was particularly costly?**
Recall, since Recall = TP/(TP+FN) directly measures how many real positive cases
are caught. Current recall is 66.67% — in a scenario where false negatives are
costly, this would need improving, commonly by lowering the classification
threshold below 0.5 (trading some precision for recall) or weighting the loss
function to penalize false negatives more heavily.

## Part 6: Training Analysis

**Loss curves:** Training and validation loss both decreased steadily and smoothly
throughout training. From roughly epoch 120 onward, validation loss stayed
consistently below training loss, and both curves flattened/plateaued by around
epoch 250-300.

**Accuracy curves:** Both train and validation accuracy stayed flat around ~38%
(near the minority class proportion) for the first ~50 epochs, indicating the
model had not yet found a useful signal. Around epoch 55-65 there was a sharp jump
to ~75-80% accuracy, consistent with the model rapidly learning to exploit a
strong feature (likely Sex, given its dominance in EDA). From roughly epoch 130
onward, validation accuracy consistently sat above training accuracy, ending at
Val 85.8% vs Train 82.0%.

**Overfitting/underfitting diagnosis:** No overfitting observed. Overfitting
specifically requires validation performance to plateau or worsen while training
performance keeps improving — instead, both loss and accuracy improved together
throughout training, with validation performing as well as or better than
training. No underfitting either, given the model clearly moved well beyond the
naive 61.6% baseline.

**Note on validation outperforming training:** This is consistent across both
loss and accuracy, and stable over 150+ epochs, not just noise. Most likely
explanation is sampling variance — the validation set is only 134 rows, so
which specific passengers land in that split by chance can meaningfully shift
its average difficulty relative to the 623-row training set. A more rigorous
setup (e.g., k-fold cross-validation) would help confirm this, and is noted as
a possible extension.

**Run-to-run variance note:** Test accuracy varied slightly between two training
runs on identical code (76.12% vs 75.37%), because neural network weights are
randomly initialized each run. A more robust evaluation would average results
across multiple training runs rather than relying on a single run.

**If overfitting had been observed, mitigation strategies would include:**
reducing model capacity (fewer neurons/layers), adding Dropout, L2 weight
regularization, early stopping based on validation loss, or gathering more
training data.

## Part 7: Experiment — Comparing Model Variations

**Setup:** Refactored the model into a parameterized FlexibleNet class, allowing
architecture/hyperparameters to be swapped via arguments. Ran the baseline plus
five variations, each changing exactly one setting relative to baseline, with a
fixed random seed (42) before each model's creation to control for random weight
initialization differences.

**Results:**

| Model                      | Accuracy | Recall | F1     |
|-----------------------------|----------|--------|--------|
| Baseline (2 layers, ReLU)   | 0.7463   | 0.6863 | 0.6731 |
| Exp A: 1 hidden layer       | 0.7910   | 0.5882 | 0.6818 |
| Exp B: Tanh activation      | 0.7687   | 0.6471 | 0.6804 |
| Exp C: lr = 0.01            | 0.7463   | 0.6471 | 0.6600 |
| Exp D: Dropout 0.3          | 0.7313   | 0.6667 | 0.6538 |
| Exp E: 32-16 neurons        | 0.7537   | 0.6471 | 0.6667 |

**Key observation:** No single model wins on every metric. Exp A (1 hidden layer)
achieves the highest accuracy (79.10%) but the lowest recall (58.82%) by a wide
margin — a clear illustration of the precision/recall tradeoff discussed in
Part 5. The simpler model appears to lean more heavily toward predicting the
majority class, inflating accuracy at the direct cost of missing real survivors.

**Model chosen: Baseline (2 hidden layers, ReLU).** Per the Part 5 Question 3
reasoning (recall should be prioritized when false negatives are costly),
Baseline has the best recall of all six models (68.63%). Its accuracy is a few
points below Exp A's, but that tradeoff is justified given recall was
established as the more important metric for this type of problem — choosing
Exp A for its higher accuracy would mean accepting significantly worse
performance on exactly the failure mode identified as costly.

**Secondary observation:** Dropout (Exp D) and increased neuron count (Exp E)
did not improve results over baseline. This is consistent with the Part 6
finding that the baseline model was not overfitting — Dropout is specifically
an overfitting mitigation technique, so its limited effect here

## Part 8: Predictions on Unseen Samples

Constructed 5 hypothetical passengers spanning a range of profiles, scaled using
the training-set-derived scaler (never refit on these 5 rows, consistent with
avoiding data leakage at inference time).

| Passenger | Profile                                          | Probability | Prediction        |
|-----------|---------------------------------------------------|--------------|--------------------|
| 1         | 1st class, female, 28, alone, high fare            | 0.9276       | Survived           |
| 2         | 3rd class, male, 22, alone, low fare               | 0.1096       | Did Not Survive    |
| 3         | 2nd class, female, 4, with family, moderate fare   | 0.9046       | Survived           |
| 4         | 3rd class, male, 35, alone, low fare, Queenstown   | 0.1336       | Did Not Survive    |
| 5         | 1st class, male, 60, with spouse, high fare        | 0.3932       | Did Not Survive    |

**Discussion:** Passengers 1-4 were predicted with high confidence (all
probabilities either >90% or <14%), consistent with EDA findings that class,
sex, and family accompaniment are strong survival predictors. Passenger 5 is
the most interesting case: unlike the other four, its probability (39.3%) sits
much closer to the 0.5 decision boundary. This passenger has competing signals
— high class and spousal accompaniment favor survival, while older age and male
sex work against it — and the model's less-confident, near-boundary output
suggests it is weighing multiple features together rather than relying on a
single dominant shortcut rule (e.g., "if male, always predict low"). This is a
more convincing sign of genuine multi-feature learning than the confident
predictions on the other four, more clear-cut passengers.

## Bonus 1: Hyperparameter Tuning

Ran a grid search over 3 learning rates (0.001, 0.01, 0.0001) x 3 architectures
(small [8], medium [16,8] = baseline, large [32,16]) = 9 total combinations,
using a fixed random seed before each model's creation for fair comparison.

**Best combination:** LR=0.001, Architecture=large [32,16] — Accuracy 78.36%,
Recall 68.63%, F1 0.7071 (best F1 across the entire project so far, edging out
the original baseline's 0.6731).

**Key finding — underfitting demonstrated directly:** LR=0.0001 produced Recall
and F1 of exactly 0.0000 for both the small and large architectures, meaning the
model predicted "died" for every single test passenger. This is a concrete,
reproducible example of underfitting: the learning rate was too small for 200
epochs to move the weights meaningfully past their random starting point, likely
never reaching the sharp "signal discovery" jump seen around epoch 55-65 in the
baseline's own accuracy curve (Part 6).

**General pattern:** LR=0.001 consistently outperformed both LR=0.01 (moderate,
less stable) and LR=0.0001 (too slow, near-collapse) across all three
architectures, validating the original baseline's learning rate choice.

## Bonus 2: Model Comparison — Neural Network vs Classical ML

Trained Logistic Regression, Decision Tree (max_depth=5), and Random Forest
(100 trees, max_depth=5) on the identical preprocessed train/test split used
for the neural network.

| Model                      | Accuracy | Recall | F1     |
|------------------------------|----------|--------|--------|
| Logistic Regression         | 0.7388   | 0.6863 | 0.6667 |
| Decision Tree                | 0.7537   | 0.5098 | 0.6118 |
| Random Forest                 | 0.7761   | 0.6078 | 0.6739 |
| Neural Network (baseline)    | 0.7463   | 0.6863 | 0.6731 |

**Key finding:** Logistic Regression — mathematically the simplest possible
model (equivalent to a neural network with zero hidden layers) — matched the
Neural Network's recall exactly (68.63%) and scored close on F1. This is strong
evidence that survival patterns in this dataset are largely linear/simple
enough that a full neural network's hidden-layer capacity for learning
nonlinear feature interactions isn't providing a decisive advantage here.

Decision Tree underperformed on recall (50.98%) relative to the ensemble
Random Forest (60.78%), illustrating why ensembling exists: a single tree is
more sensitive/unstable to which specific rows land in train vs test on a
small dataset, and Random Forest's averaging over many trees smooths this out.

**Conclusion:** Given recall was established as the priority metric (Part 5,
Q3), Logistic Regression and the Neural Network are effectively tied as the
best choices, with Logistic Regression achieving this with far less complexity
and computational cost. The neural network remains a valid, competitive choice,
and fulfills the task's explicit goal of demonstrating NN fundamentals, but
this comparison shows it is not strictly necessary for this dataset.

## Bonus 3: Explainability — Permutation Feature Importance

Used permutation importance on the trained (saved/reloaded) baseline neural
network: for each feature, shuffled its values in the test set (breaking its
relationship to the target while keeping all other features intact), and
measured the resulting drop in test accuracy relative to baseline (76.12%).

| Feature     | Accuracy after shuffle | Importance (accuracy drop) |
|--------------|--------------------------|------------------------------|
| Sex          | 0.5672                   | 0.1940                        |
| SibSp        | 0.7164                   | 0.0448                        |
| Age          | 0.7239                   | 0.0373                        |
| Parch        | 0.7463                   | 0.0149                        |
| Pclass       | 0.7537                   | 0.0075                        |
| Embarked_Q   | 0.7537                   | 0.0075                        |
| Fare         | 0.7687                   | -0.0075                       |
| Embarked_S   | 0.7687                   | -0.0075                       |

**Key finding:** Sex is overwhelmingly the most important feature — shuffling
it alone drops accuracy by 19.4 percentage points, roughly 4-5x larger than
the next most important features (SibSp, Age). This directly confirms the EDA
observation from Part 2 (female survival ~74% vs male ~19%) using an
independent, model-based method rather than just visual inspection.

Fare and Embarked_S showed small negative importance values, which is
attributable to sampling noise given the small (134-row) test set — a
single flipped prediction is worth ~0.0075 in accuracy, matching the
magnitude seen here — rather than a genuine negative effect. A more robust
estimate would average multiple permutation repeats per feature.

**Connection to Bonus 2:** Sex's dominant, near-linear relationship with
survival helps explain why Logistic Regression performed comparably to the
Neural Network — when one feature carries most of the signal in a fairly
direct way, a simple linear model can capture much of that relationship
without needing hidden-layer nonlinearity.