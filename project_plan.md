# Project Plan: Predicting Grocery Reorders

**A Comparison of ML Classification Methods on the Instacart Market Basket Analysis Dataset**

---

## 1. Project Overview

We frame grocery reorder prediction as a binary classification problem: for each (user, product) pair where the user has previously purchased the product, predict whether it will appear in the user's next order. We compare four classical models — Logistic Regression, Random Forest, AdaBoost, and XGBoost — against a simple counting baseline, with the goal of analyzing where each model succeeds and fails rather than chasing leaderboard performance.

The project unfolds in five phases:

1. **Exploratory data analysis (EDA)** on the Instacart dataset
2. **Feature engineering** — compute 44 candidate features drawn from prior public implementations, apply a logical pre-filter, then a correlation diagnostic to control multicollinearity (especially for logistic regression)
3. **Preprocessing pipeline** with strict data leakage controls
4. **Model training and tuning** — grid search for logistic regression, randomized search for the rest
5. **Comparative error analysis** across model types to characterize each model's strengths and weaknesses

---

## 2. Phase 1: Exploratory Data Analysis

Before any feature engineering, we conduct EDA to understand the data we're working with. This both informs feature selection (Phase 2) and surfaces issues that would silently break a downstream pipeline (e.g., the 30-day cap on `days_since_prior_order`, noted by Damang in `supermarket_sweep`).

### 2.1 Dataset summary statistics

We start by loading the five CSVs (`orders`, `order_products__prior`, `order_products__train`, `products`, `aisles`, `departments`) and computing:

- Total counts (~206K users, ~3.4M orders, ~50K products)
- Distribution of orders per user (range 4–100 per Instacart's documentation)
- Distribution of products per order (basket size)
- Distribution of `days_since_prior_order` — expecting a spike at 30 due to Instacart's cap
- Day-of-week and hour-of-day distributions for orders
- Reorder rate overall and by aisle/department

### 2.2 Target variable analysis

For each (user, product) pair where the user has purchased the product before, we compute the binary label (was the product in the user's final order).

**Important note on class imbalance.** The widely cited "59% reorder rate" from Instacart's documentation describes the fraction of items in a typical order that are reorders, *not* the positive class rate of our binary classification problem. Our problem operates over a different denominator: every labeled user's historical (user, product) pair is a candidate, and only the small subset that actually appears in the user's target order is positive. On the official Kaggle train users, this yields 828,824 positives out of 8,474,661 candidate pairs, or **9.78% positive**. If the candidate table is built over all users, including official Kaggle test users whose labels are unavailable, the apparent rate falls to about 6.22%; that is a denominator error because unlabeled test users have been filled as negatives.

The expected positive class rate for our prediction task is therefore approximately **9.8% positive / 90.2% negative** — heavy class imbalance, not the mild imbalance suggested by the original 59/41 figure. This shapes several downstream choices in this plan:

- Accuracy is unsuitable as a primary metric (a constant "no" predictor would score ~90%); we use F1-score as primary instead (Section 6.1).
- Threshold tuning sweeps target F1, not accuracy (Section 5.4).
- Per-row F1 versus per-order F1 must be specified explicitly (Section 6.1).

We verify the empirical imbalance during EDA and check whether it varies meaningfully by:

- User activity level (do power users have more positives?)
- Product popularity (are popular products easier to predict?)
- Department (produce vs. household goods)

### 2.3 Cold-start and edge cases

We identify the populations our error analysis (Phase 5) will examine:

- Users with very few prior orders (<5)
- Products purchased only once by a user (single-purchase products)
- Users whose final order contains no reorders ("None" cases)
- High-variance categories where reorder rates vary widely

---

## 3. Phase 2: Feature Engineering

### 3.1 Approach: shared feature set with embedded selection (Option A)

Rather than aggressively pre-selecting features per model, we adopt a **shared feature set** strategy. All four models are trained on the same engineered feature set, and per-model feature selection happens implicitly through each model's native regularization or splitting mechanism (L1 penalty for logistic regression, `gamma` and column subsampling for XGBoost, stump-based splitting for AdaBoost, `max_features` for Random Forest).

This choice is deliberate. Our project goal is to **compare model classes and analyze where each succeeds and fails on a common task** — not to maximize raw accuracy on a per-model leaderboard. Holding the input feature set constant is the experimentally clean way to attribute performance differences to modeling choices rather than to differences in input. It also enables the cross-model feature importance analysis in Phase 5 — comparisons like "XGBoost ranks `up_purchase_count_n5` first while AdaBoost ignores it" only make sense if both models had access to the feature.

The shared feature set is constructed in two pre-training steps:

1. **Logical pre-filter (Section 3.3)** — drop features that are redundant or derivable from others, based on reasoning about feature definitions. ~8 features removed.
2. **Correlation diagnostics (Section 3.4)** — compute the full feature correlation matrix; investigate any pair with `|r| > 0.9` and drop one if multicollinearity is confirmed. This step is especially important for logistic regression, whose coefficients become unstable and L1 selection becomes erratic when correlated features compete for the same regularization budget.

We accept a possible ~0.005–0.01 accuracy cost relative to per-model feature selection. Public implementations on this dataset (Olson-Swanson 2018; Yu 2017) report this is the typical scale of gain from per-model feature pruning, which is well within noise on our test set.

### 3.2 The 44-feature candidate pool

The candidate features are drawn from public implementations of this prediction task. The authoritative sources are:

- **Onodera, K. (2017)** — 2nd place in the Kaggle Instacart Market Basket Analysis competition. His writeup documents the four-group taxonomy (User / Product / User×Product / Datetime) we adopt. His `_n5` recency-windowed features are the strongest predictors in his model.
- **Olson-Swanson, A. (2018)** — Documents specific F1 progression as features are added; reports that `days_since_user_last_bought_this_product` alone moved logistic regression F1 from 0.39 to 0.42, and last-3-orders features moved it further to 0.43.
- **Vasquez, S. (2017)** — 3rd place solution; uses many of the same direct aggregations alongside more advanced learned features (which we exclude — see Section 3.4).
- **Rich, A. (2017)** — 23rd place; documents recency-decayed predictors and confirms direct aggregations dominate.
- **syntheticjohn (2019)** — Reports XGBoost F1 = 0.440 using a similar feature set.
- **Yu, D. (2017)** — Reports leaderboard F1 jumping from 0.32 to 0.38 from feature additions alone (vs. minor gains from hyperparameter tuning).
- **Bhat, A. (2022)** — Builds 76 features and reports their importance rankings.

The 44 candidate features below are organized into four groups, with each feature attributed to its primary source(s) in the literature. **All 44 features enter the data-driven pre-filter in Section 3.3.** No upfront subjective drops.

#### Group 1: User-level features (12 candidates)

| # | Feature | Definition | Source | Pre-filter decision |
|---|---|---|---|---|
| 1 | `user_total_orders` | Number of prior orders by the user | Onodera, Olson-Swanson, Bhat | Keep |
| 2 | `user_total_products` | Total products across all prior orders | Bhat | Keep |
| 3 | `user_total_distinct_products` | Unique products ever purchased | Bhat | Keep |
| 4 | `user_avg_basket_size` | Mean products per order | Onodera, Olson-Swanson | Keep |
| 5 | `user_std_basket_size` | Std dev of basket size | Bhat | Keep |
| 6 | `user_avg_days_between_orders` | Mean inter-order gap | Onodera, Olson-Swanson, Bhat | Keep |
| 7 | `user_std_days_between_orders` | Std dev of inter-order gap | Bhat | **Drop** — strongly correlated with mean for users with regular cadence; adds little |
| 8 | `user_reorder_ratio` | Fraction of all purchases that were reorders | Onodera (`t-1_reordered_ratio`) | Keep |
| 9 | `user_unique_aisles` | Distinct aisles shopped | Bhat | **Drop** — bounded by `user_total_distinct_products`; near-perfect correlation expected |
| 10 | `user_unique_departments` | Distinct departments shopped | Bhat | **Drop** — same reason as #9 (only 21 departments exist; bounded ceiling) |
| 11 | `user_pref_dow` | Mode day-of-week | Olson-Swanson | Keep |
| 12 | `user_pref_hour` | Mode hour-of-day | Olson-Swanson | Keep |

**Group 1: 9 features kept, 3 dropped.**

#### Group 2: Product-level features (10 candidates)

| # | Feature | Definition | Source | Pre-filter decision |
|---|---|---|---|---|
| 13 | `prod_total_purchases` | Times the product has been purchased globally | Onodera, Olson-Swanson | Keep |
| 14 | `prod_total_users` | Distinct users who bought it | Bhat | **Drop** — for popular products (most of the signal) this nearly equals `prod_total_purchases` divided by mean purchases per user; collinear |
| 15 | `prod_reorder_ratio` | Fraction of purchases that were reorders | Onodera, Vasquez | Keep |
| 16 | `prod_reorder_probability` | P(buy again \| bought once) — fraction of users who bought ≥2 times among those who bought ≥1 time | Onodera ("probability reordered after first order") | Keep |
| 17 | `prod_avg_cart_position` | Mean `add_to_cart_order` | Onodera | Keep |
| 18 | `prod_avg_orders_between` | Mean orders between consecutive purchases | Onodera | Keep |
| 19 | `aisle_id` | Categorical (134 values) | All | Keep (label-encoded for trees; replaced by `aisle_reorder_ratio` for LR) |
| 20 | `department_id` | Categorical (21 values) | All | Keep (same handling) |
| 21 | `aisle_reorder_ratio` | Mean reorder rate of products in aisle | Bhat | Keep |
| 22 | `dept_reorder_ratio` | Mean reorder rate of products in department | Bhat | Keep |

**Group 2: 9 features kept, 1 dropped.**

#### Group 3: User × Product features — base (11 candidates)

This is the most signal-rich group. Onodera's top-5 reorder predictors are all from this group, and Olson-Swanson's largest single F1 jump came from feature #25 below.

| # | Feature | Definition | Source | Pre-filter decision |
|---|---|---|---|---|
| 23 | `up_purchase_count` | Times user bought the product | All | Keep |
| 24 | `up_first_order_number` | Order # of first purchase | Onodera | Keep |
| 25 | `up_last_order_number` | Order # of most recent purchase | Onodera | Keep |
| 26 | `up_orders_since_last` | Orders elapsed since last purchase | Onodera | Keep |
| 27 | `up_days_since_last` | Actual days elapsed since last purchase | Olson-Swanson (top contributor) | Keep |
| 28 | `up_order_ratio_by_chance` | Purchase count / orders since first encounter | Onodera (top-5) | Keep |
| 29 | `up_avg_cart_position` | Mean cart position for this user-product | Onodera | Keep |
| 30 | `up_avg_days_between` | Mean days between user's purchases of this product | Onodera | Keep |
| 31 | `up_max_days_between` | Longest gap between purchases | Onodera (top-5: `useritem_order_days_max`) | Keep |
| 32 | `up_streak` | Consecutive most-recent orders containing the product | Onodera, Vasquez | Keep |
| 33 | `up_reorder_rate_after_first` | Fraction of post-first orders containing the product | Bhat | Keep |

**Group 3: all 11 kept.** Every feature is documented as influential by at least one source, and several are among Onodera's top predictors. We keep `up_orders_since_last` and `up_days_since_last` despite their conceptual overlap — they correlate but not perfectly (variation in inter-order gap means orders-since and days-since carry different information for irregular users), and they will be cross-checked in the correlation diagnostic step (3.4).

#### Group 3b: User × Product features — recency-windowed (5 candidates)

Onodera's top features all carry the `_n5` suffix (computed over the user's last 5 orders only). Olson-Swanson found that switching to last-3-orders versions of features improved his F1 from 0.42 to 0.43. We adopt the `_n5` window.

| # | Feature | Definition | Source | Pre-filter decision |
|---|---|---|---|---|
| 34 | `up_purchase_count_n5` | Times bought in last 5 orders (0–5) | Onodera (top-1) | Keep |
| 35 | `up_purchase_ratio_n5` | `up_purchase_count_n5 / 5` | Onodera (top-2) | **Drop** — exact linear transformation of #34 (`x/5`); deterministic redundancy |
| 36 | `up_orders_since_last_n5` | Orders since last purchase, capped at 5 | Onodera | Keep |
| 37 | `up_avg_cart_position_n5` | Mean cart position in last 5 orders | Onodera | Keep |
| 38 | `up_streak_n5` | Streak within last 5 orders | Onodera | Keep |

**Group 3b: 4 features kept, 1 dropped.** We drop `up_purchase_ratio_n5` because it is a perfect linear transformation of `up_purchase_count_n5` — they will have correlation = 1.0 by construction, which is harmful for logistic regression. We keep `up_avg_cart_position_n5` and `up_streak_n5` despite some overlap with their full-history counterparts (#29, #32) — the recency-windowed versions carry distinct signal for users whose behavior changed recently. The correlation matrix in Section 3.4 will verify this.

#### Group 4: Datetime features (6 candidates)

`orders.csv` provides the day-of-week, hour, and `days_since_prior_order` for the *target order* (the one we're predicting). This is free signal Onodera and Olson-Swanson both use.

| # | Feature | Definition | Source | Pre-filter decision |
|---|---|---|---|---|
| 39 | `target_dow` | Day-of-week of target order | Onodera | Keep |
| 40 | `target_hour` | Hour of target order | Onodera | Keep |
| 41 | `target_days_since_prior` | Days since user's previous order | Onodera, Olson-Swanson | Keep |
| 42 | `dow_match_user_pref` | 1 if `target_dow == user_pref_dow` | Bhat | **Drop** — derivable as a deterministic function of #11 and #39; tree models can learn the interaction natively |
| 43 | `hour_diff_from_user_pref` | `\|target_hour − user_pref_hour\|` | Bhat | **Drop** — derivable from #12 and #40; same reason |
| 44 | `up_dow_purchase_count` | Times user bought this product on this DOW | Bhat | Keep — three-way interaction (user × product × DOW) not derivable from base features |

**Group 4: 4 features kept, 2 dropped.** The dropped interactions are deterministic functions of features that are kept; they would inflate the feature count without adding information. We retain `up_dow_purchase_count` because it captures a three-way interaction that no single base feature encodes.

### 3.3 Pre-filter summary: 37 features after logical pre-filter

| Group | Candidates | After pre-filter | Dropped |
|---|---|---|---|
| User-level | 12 | 9 | 3 |
| Product-level | 10 | 9 | 1 |
| User × Product (base) | 11 | 11 | 0 |
| User × Product (recency `_n5`) | 5 | 4 | 1 |
| Datetime | 6 | 4 | 2 |
| **Total** | **44** | **37** | **7** |

We carry 37 features into the correlation diagnostic (Section 3.4). Any further drops there are based on measured multicollinearity rather than reasoning about feature definitions.

### 3.4 Correlation matrix and multicollinearity check

The logical pre-filter handles redundancies that are obvious from feature definitions, but it cannot catch empirical multicollinearity — features that are definitionally distinct but turn out to be highly correlated in practice (e.g., `up_orders_since_last` vs. `up_days_since_last` for users with regular order cadence).

Multicollinearity matters specifically for **logistic regression**:

- Standardized coefficients become unstable: small changes in training data swing coefficients between correlated features.
- L1 selection becomes erratic: the penalty arbitrarily picks one of several correlated features to zero out, making the "selected" feature set non-reproducible across runs.
- Coefficient signs may flip between training folds, breaking interpretability.

Tree-based models (Random Forest, AdaBoost, XGBoost) are **mostly robust** to multicollinearity — they can split on either of two correlated features and reach equivalent predictions — but tree-based feature importances are split between correlated features, making per-feature importance harder to interpret. So multicollinearity matters for our analysis even when it doesn't hurt accuracy.

**Procedure:**

1. **Compute the full Pearson correlation matrix** for the 37 numerical features on the training fold.
2. **Visualize as a heatmap** ordered by feature group, included in the EDA report.
3. **Flag pairs with `|r| > 0.9`** for review.
4. **For each flagged pair:** drop the feature that is more derivative (e.g., a ratio constructed from another kept feature) or less commonly cited in prior implementations.
5. **For aisle/department features used in logistic regression:** as discussed in Section 3.5, raw `aisle_id` and `department_id` are dropped for LR in favor of the target-encoded `aisle_reorder_ratio` and `dept_reorder_ratio`.
6. **Variance Inflation Factor (VIF) spot check.** For the surviving features, compute VIF using `statsmodels.stats.outliers_influence.variance_inflation_factor`. We use **VIF > 5** as the threshold for investigation rather than the looser conventional VIF > 10 — because logistic regression coefficient interpretability is part of our analysis (Phase 5 reports standardized coefficients as feature importance), and the tighter threshold is a more rigorous standard when interpretation matters (Belsley, Kuh & Welsch, 1980). Features exceeding VIF > 5 are reviewed and dropped if no clear retention rationale exists.

**Expected outcome.** Based on feature definitions, we expect to flag and investigate roughly 3–5 pairs:

| Likely flagged pair | Expected `|r|` | Likely action |
|---|---|---|
| `up_orders_since_last` vs. `up_days_since_last` | ~0.85–0.90 | Borderline; likely keep both, but verify VIF |
| `prod_total_purchases` vs. `prod_reorder_ratio × prod_total_purchases` | varies | Verify; should be uncorrelated by construction |
| `user_total_orders` vs. `user_total_products` | ~0.7–0.85 | Likely keep both; correlation captures different signal |
| `up_streak` vs. `up_streak_n5` | ~0.6–0.8 | Keep both unless > 0.9 |
| `up_avg_cart_position` vs. `up_avg_cart_position_n5` | ~0.6–0.8 | Keep both unless > 0.9 |

The exact set of dropped features depends on measurements; we estimate **2–4 additional drops** beyond the logical pre-filter, leaving **~33–35 features** going into model training.

The correlation matrix is computed and any drop decisions are made **on the training fold only** during cross-validation, to avoid leaking information from validation data into the feature set (Section 4.3).

### 3.5 What we explicitly exclude

Some features used by top finishers are deliberately out of scope:

- **Word2vec / GloVe product embeddings** (Rich, Pan) — adds a second pipeline (training an embedding model first); marginal documented gains.
- **NNMF / matrix factorization features** (Vasquez) — same concern.
- **KNN / collaborative-filtering similarity features** — memory-intensive and blurs the line between "feature engineering" and "modeling."
- **Product co-occurrence and replacement features** (Onodera) — requires building a 50K × 50K co-occurrence matrix; high implementation cost for marginal gain.
- **Per-order F1-threshold optimization** (Onodera) — orthogonal to our model-comparison framing; mentioned in future work.

These are mentioned in the proposal's "Limitations and Future Work" section so reviewers know we're aware of them.

### 3.6 Per-model preprocessing

All four models receive **the same set of features** (those surviving the Section 3.3 pre-filter). Per-model differences are limited to mechanical preprocessing — scaling, categorical encoding, and missing-value handling — that reflects each model class's mathematical requirements, not feature selection.

| Step | Logistic Regression | Random Forest | AdaBoost | XGBoost |
|---|---|---|---|---|
| Numeric scaling | Min-max scale | None | None | None |
| Categorical (`aisle_id`, `dept_id`) | Drop raw IDs; rely on `aisle_reorder_ratio` and `dept_reorder_ratio` (target-encoded equivalents) | Label-encode | Label-encode | Label-encode (or `enable_categorical=True`) |
| Missing values | Median impute | Median impute | Median impute | Leave as NaN (XGBoost handles natively) |
| Class imbalance | Threshold tuning on validation set | Threshold tuning | Threshold tuning | Threshold tuning |

Note on AdaBoost: scikit-learn's `AdaBoostClassifier` uses decision tree stumps by default and is closer to a tree model than a linear one — we treat it like Random Forest for preprocessing purposes.

**Embedded feature selection happens during model training, not as a preprocessing step.** Specifically:

- **Logistic regression** with L1 penalty (tuned via grid search over `C`) drives unhelpful coefficients to exactly zero, performing built-in feature selection.
- **Random Forest** with tuned `max_features` and tree-based splitting ignores unhelpful features by simply not splitting on them.
- **AdaBoost** with stumps (`max_depth=1`) naturally concentrates weight on a small subset of features — typically 20–30 unique features even after 100 boosting rounds.
- **XGBoost** with tuned `gamma` (minimum loss to split), `min_child_weight`, and `colsample_bytree` effectively prunes weak splits and ignores unhelpful features during training.

This means each model is implicitly picking its own "preferred" feature subset from the shared input, without us imposing per-model feature drops upfront. The result is a clean experimental setup (same inputs, fair comparison) that still allows each model to use only what it finds useful.

---

## 4. Phase 3: Data Splitting and Leakage Controls

Data leakage is a real risk in this project because of how features are constructed. The user-level and product-level aggregates are computed across the prior dataset, and if they're computed before splitting, validation/test labels can leak into them.

### 4.1 Train-test split: repurposing the Kaggle train set

Because the ground-truth labels for the official Kaggle test split remain private, we discarded the Kaggle test set for this study. Instead, we repurposed the official Kaggle train set — which contains the fully labeled "next orders" for 131,209 users — as our complete ground-truth dataset. To rigorously evaluate our models, we performed a custom **80/20 user-level stratified split** on this data, yielding a **Local Training set (~105k users)** for model fitting and hyperparameter tuning, and a **Local Test set (~26k users)** strictly held out for final evaluation.

The split is performed **at the user level**, not by row. If a user's (user, banana) row went to train and (user, apple) to test, the user-level features (`user_total_orders`, `user_pref_hour`, etc.) — which are computed identically for both rows — would leak the user's behavior across the split. By assigning each user entirely to one side, this is avoided.

Stratification is by user activity tier (low / medium / high `user_total_orders`) so that all activity levels appear in both sets in proportions matching the source data.

### 4.2 Cross-validation: 5-fold, user-grouped

Within the training set, we use 5-fold cross-validation with `GroupKFold` on `user_id`. This guarantees that no user appears in both train and validation folds during hyperparameter tuning.

### 4.3 Leakage-safe feature computation

The four specific risks:

1. **Target encoding leakage and smoothing.** `aisle_reorder_ratio` and `dept_reorder_ratio` are computed from labels. If computed on the full dataset before splitting, training is inflated by validation labels. Mitigation: compute these aggregates **inside each CV fold** using only that fold's training data, then merge into the validation rows. Additionally, raw mean-target encoding is unstable for categories with few observations — an aisle with only 5 purchases will have a noisy reorder ratio. We therefore apply **Laplace (additive) smoothing** following Micci-Barreca (2001), which blends each category's empirical mean with the global mean, weighted by category sample size:

   $$\hat{x}_l = \frac{n_l \cdot \bar{y}_l + m \cdot \bar{y}_{\text{global}}}{n_l + m}$$

   where $n_l$ is the count of training rows for category $l$, $\bar{y}_l$ is the in-category target mean, $\bar{y}_{\text{global}}$ is the overall target mean, and $m$ is the smoothing strength (we use $m = 10$ as a moderate prior). This both reduces variance for rare categories and provides a sensible default for unseen categories at test time.

2. **Future-information leakage in user/product features.** All aggregations must use only the user's *prior* orders — never the target order. We verify this by ensuring the label-providing order is excluded from all aggregation computations (it is, by construction in the Instacart data: `prior` orders are separate from the `train` order).

3. **Correlation-driven drop leakage.** The correlation matrix (Section 3.4) and any drops based on it must be computed **on the training fold only** during cross-validation. The same drop list is then applied to validation and test data. If correlations were computed on the full dataset (training + validation), validation information would leak into the feature-selection decision.

4. **Imputation leakage.** Median imputation values are fit on the training fold only and applied to the validation fold.

The logical pre-filter (Section 3.3) does not introduce leakage — it makes drops based on feature definitions, not on data. Those drops are applied uniformly across all folds.

### 4.4 Verification step

Before any model training, we run a sanity check: train a logistic regression on the features and evaluate on validation. If validation F1 is suspiciously close to training F1 (e.g., difference < 0.01), or if any feature has a univariate correlation > 0.95 with the label, we investigate for leakage.

---

## 5. Phase 4: Models and Hyperparameter Tuning

### 5.1 Models

| Model | Role | Why it's in the comparison |
|---|---|---|
| Frequency baseline | Simple counting rule (predict reorder if user purchase count > θ) | Establishes a "no recency, only frequency" floor |
| Last-basket baseline | Predict the user will reorder exactly the items in their previous order | Strong recency-only heuristic; famously hard to beat in next-basket recommendation |
| Logistic Regression | Linear baseline | Fast, interpretable, fundamentally different from trees |
| Random Forest | Bagged trees | Independent ensemble; robust to overfitting |
| AdaBoost | Sequential boosted stumps (`SAMME` algorithm; sklearn deprecated `SAMME.R` in v1.4+) | Simpler boosting algorithm; contrasts with XGBoost |
| XGBoost | Gradient-boosted trees | State-of-the-art for this dataset (2nd–3rd place winners used XGBoost/LightGBM) |

The two baselines are complementary: the frequency baseline isolates "how much does buying a product often predict reordering it?" while the last-basket baseline isolates "how much does recent purchase predict reordering?" If our ML models cannot beat the last-basket baseline, they are not adding value beyond a one-line heuristic — this is a meaningful sanity check borrowed from the next-basket recommendation literature.

For AdaBoost, we use the discrete `SAMME` algorithm rather than `SAMME.R`, because scikit-learn 1.4+ deprecated `SAMME.R`. This ensures forward compatibility and reproducibility.

### 5.2 Tuning strategy by model

Following the practice of public Instacart implementations (Onodera 2017; Olson-Swanson 2018; Yu 2017), we allocate tuning effort based on model complexity and fit time.

#### 5.2.1 Frequency baseline

Single threshold `θ` swept from 1 to 20 on validation; predict reorder if the user's purchase count for this product exceeds θ. No CV needed.

#### 5.2.2 Last-basket baseline

No tuning. The prediction rule is deterministic: for each (user, product) pair, predict 1 if the product appeared in the user's most recent prior order, else 0. We compute its F1, precision, recall, and PR-AUC on the test set as a strong recency-only reference point.

#### 5.2.3 Logistic regression — full grid search

Logistic regression fits quickly, so a full grid search is computationally feasible and provides the cleanest tuning report.

```
penalty: ['l1', 'l2']
C:        [0.001, 0.01, 0.1, 1, 10, 100]
solver:   'saga'  (handles both L1 and L2)
class_weight: [None, 'balanced']
```

That's 2 × 6 × 2 = **24 combinations × 5 folds = 120 fits**. Tractable.

Implemented with `GridSearchCV(scoring='accuracy', cv=GroupKFold(n_splits=5))`.

#### 5.2.4 Random Forest — randomized search

Random Forest has more hyperparameters and longer fit times. We use `RandomizedSearchCV` with 30 samples.

```
n_estimators:        [100, 200, 300, 500]
max_depth:           [None, 10, 15, 20, 30]
min_samples_split:   [2, 5, 10]
min_samples_leaf:    [1, 2, 4]
max_features:        ['sqrt', 'log2', 0.5]
class_weight:        [None, 'balanced']
```

Theoretical grid: 4 × 5 × 3 × 3 × 3 × 2 = 1,080 combinations. We sample 30, which prior research (Bergstra & Bengio, 2012) shows is sufficient to match grid-search-quality tuning when most hyperparameters are not equally important.

#### 5.2.5 AdaBoost — randomized search

```
algorithm:       'SAMME'  (fixed; SAMME.R deprecated in sklearn 1.4+)
n_estimators:    [50, 100, 200, 300]
learning_rate:   [0.01, 0.05, 0.1, 0.5, 1.0]
estimator:       DecisionTreeClassifier(max_depth=[1, 2, 3])
```

Sample 20 random combinations.

#### 5.2.6 XGBoost — Optuna with per-fold early stopping

XGBoost requires special handling. The naive approach — using `RandomizedSearchCV` with `early_stopping_rounds` and a global `eval_set` — is **broken**: every CV fold uses the same global validation data to trigger early stopping, leaking validation information into hyperparameter selection. Without an `eval_set`, early stopping doesn't work at all. We avoid this with a custom Optuna trial loop that performs early stopping correctly inside each fold.

For each Optuna trial, the procedure is:

1. Sample a hyperparameter configuration from the search space.
2. For each of the 5 user-grouped CV folds:
   a. Within the training fold, hold out an inner validation slice (10% of fold-training users) for early stopping.
   b. Fit XGBoost with `n_estimators=2000` and `early_stopping_rounds=50` on the inner training data, monitoring inner validation F1.
   c. Predict on the outer validation fold (which the trial has never seen) and record F1.
3. Mean F1 across the 5 folds is the trial's objective value.

Search space:

```
learning_rate:    [0.01, 0.05, 0.1, 0.2]   (log-uniform sampling)
max_depth:        [4, 6, 8, 10]
min_child_weight: [1, 5, 10]
subsample:        [0.7, 0.8, 0.9, 1.0]
colsample_bytree: [0.7, 0.8, 0.9, 1.0]
gamma:            [0, 0.1, 0.5]
```

We run **30 Optuna trials** with the TPE (Tree-structured Parzen Estimator) sampler. This is slower than `RandomizedSearchCV` but produces a correctly cross-validated result without leakage. Onodera's reported settings (`learning_rate=0.01`, `max_depth=6`, `min_child_weight=10`) fall inside these ranges, providing a sanity check that our search space is reasonable.

### 5.3 Subsampling for tuning

Following Olson-Swanson and syntheticjohn, we tune on a stratified **20% user sample** (~40K users) to keep tuning runtimes manageable, then refit the best hyperparameters on the full training set for final evaluation. This is standard practice on this dataset.

### 5.4 Threshold tuning

Default 0.5 thresholds are wrong here due to the heavy class imbalance (~10% positive). Olson-Swanson reports thresholds of 0.16 for logistic regression and 0.21 for XGBoost as F1-optimal on this dataset. After hyperparameter tuning, we sweep the decision threshold from 0.05 to 0.50 in 0.01 increments on the validation set and select the threshold **maximizing F1-score** (our primary metric — see Section 6.1), separately for each model.

---

## 6. Phase 5: Evaluation and Comparative Error Analysis

### 6.1 Metrics

**Primary metric: F1-score.** Reported on the held-out test set with each model's F1-optimal threshold (selected per Section 5.4). F1 is appropriate because it is the harmonic mean of precision and recall, suitable for the heavy class imbalance (~10% positive), and consistent with the original Kaggle competition metric for direct comparability with prior work.

**F1 aggregation method.** The literature uses two F1 conventions on this dataset, and we explicitly specify ours:

- **Per-row (global) F1** — computed once over all (user, product) test rows. This treats every prediction equally regardless of which user it belongs to.
- **Per-order (basket-level) F1** — computed separately for each user's next order, then averaged across users. This is the convention used in the Kaggle competition and most next-basket recommendation literature (Onodera 2017).

**We report both**, with **per-order F1 as the headline number** for direct comparability with prior published results on this dataset. Per-row F1 serves as a secondary metric that is more sensitive to power users (users with more candidate products contribute more rows). Differences between the two metrics are themselves diagnostic: a model that does well per-row but poorly per-order is over-predicting for high-volume users.

**Supplementary metrics:** precision, recall, PR-AUC, ROC-AUC, and accuracy.

PR-AUC is the most informative of the supplementary metrics under heavy imbalance because precision-recall curves are not inflated by abundant true negatives — a constant "no" predictor with ~90% accuracy has a PR-AUC near the positive class rate (~0.10), correctly identifying that the model has no discriminative power. ROC-AUC is threshold-independent and complements F1 by measuring raw ranking ability. Accuracy is reported only for completeness; given the class imbalance, it is uninformative as a comparative metric and we explicitly do not use it for tuning or model selection.

### 6.2 Comparative error analysis

This is the analytical core of the project — not just "which model has the highest accuracy," but **where each model fails differently**. We do four analyses:

#### 6.2.1 Confusion-matrix decomposition

For each model, report the full confusion matrix and the FP and FN rates. Key questions:

- Does logistic regression have systematically higher FN (missed reorders) than tree models?
- Does AdaBoost have higher FP (predicting reorders that didn't happen) than Random Forest?
- Where do XGBoost's errors fall — randomly distributed or concentrated in specific subgroups?

#### 6.2.2 Error-overlap analysis

For each pair of models, compute the fraction of test examples on which both models err vs. only one errs. Models that make *different* errors are candidates for ensembling; models that make the *same* errors share a blind spot. We report a 4×4 model-disagreement matrix.

#### 6.2.3 Subgroup analysis

We slice test errors by the EDA subgroups identified in Phase 1:

| Subgroup | Hypothesis |
|---|---|
| Cold-start users (<5 prior orders) | All models struggle; logistic regression may do relatively better because it has fewer parameters to overfit |
| Single-purchase products | All models struggle; Random Forest may fall back on product-level features |
| High-frequency staples (e.g., bananas, milk) | All models do well; differences are small |
| High-variance categories (snacks, treats) | XGBoost expected to do best due to capacity to model interactions |
| "None" cases (next order has no reorders at all) | All models likely overpredict reorders |

#### 6.2.4 Feature-importance comparison (the embedded-selection analysis)

This is the analytical payoff of the Option A design. Because all four models were trained on the same input feature set, differences in which features each model relies on directly reflect *how that model class extracts signal* — not differences in input.

**Step 1: Per-model importance extraction.**

| Model | Importance metric |
|---|---|
| Logistic regression | Magnitude of standardized coefficients (and which are exactly zero under L1) |
| Random Forest | Mean decrease in impurity (Gini) |
| AdaBoost | Weighted feature importance (`feature_importances_`) |
| XGBoost | Gain-based importance |

We additionally compute **permutation importance** for all four models. Permutation importance is more reliable than the model's built-in importance because it measures the actual contribution to held-out performance rather than the contribution during training. Reporting both built-in and permutation importance lets us flag features that look important during training but don't help on test.

**Step 2: Cross-model importance overlap.** For each pair of models, compute the fraction of their top-10 features that are shared. High overlap = the signal is consistent across model classes (suggests a mostly linear / additive problem). Low overlap = different models exploit different patterns. We report the resulting 4×4 overlap matrix.

**Step 3: Disagreement features.** Identify features that one model ranks highly while another ignores. The most informative comparison is **AdaBoost vs. XGBoost** (both boosting algorithms, but AdaBoost uses stumps while XGBoost uses deeper trees):

- Features ranked high by both = strong marginal signal, captured even without interactions.
- Features ranked high by XGBoost but ignored by AdaBoost = likely contributing through interactions that stumps cannot capture.
- Features ranked high by AdaBoost but not by XGBoost = should be rare; would warrant investigation.

We also compare logistic regression's L1-zeroed features against XGBoost's bottom-importance features. Strong overlap suggests both methods agree on which features are uninformative; weak overlap suggests linear and tree models disagree about what matters.

**Step 4: Drop-one ablation on key features.** Take the 5 features that appear in the top-10 for at least 3 of 4 models. For each such feature, retrain each model with that feature removed and measure the change in test accuracy. This tells us:

- Which features are *essential* (large drop when removed)
- Which are *redundant with other features* (small or zero drop because other features carry the same signal)

**Step 5: Group-level ablation.** A coarser ablation: retrain each model with each *feature group* removed (User-level / Product-level / User×Product / User×Product `_n5` / Datetime). Reports which groups carry the most weight for each model. Onodera and Olson-Swanson informally did this; we do it formally as a clean ablation table.

The combined output of Steps 1–5 is the project's central claim about feature value across model classes. We report this as the main finding alongside accuracy comparisons.

### 6.3 Statistical significance

Different metrics require different significance tests. We use the appropriate test for each:

**For per-row prediction agreement (McNemar's test).** We use McNemar's test on paired binary predictions for each model pair. McNemar tests whether the disagreement counts (Model A right & Model B wrong vs. Model A wrong & Model B right) differ significantly from a 50/50 split. This is the right test for "do these models make systematically different decisions on the same examples" — note that McNemar tests **prediction agreement at the row level**, which is closely related to but not identical to accuracy difference. With 6 model pairs, we apply Bonferroni correction (significance threshold p < 0.05/6 ≈ 0.0083).

**For F1 differences (bootstrap resampling).** McNemar does not directly test F1 differences. To compare per-order F1 between two models, we use **bootstrap resampling**: resample users with replacement 1000 times, recompute per-order F1 for both models on each bootstrap sample, and report the 95% confidence interval for the F1 difference. If the interval excludes zero, the difference is significant.

**For AUC differences (DeLong's test).** ROC-AUC and PR-AUC differences are tested using DeLong's test (DeLong et al., 1988), which accounts for the paired nature of the comparison and is implemented in `scipy.stats` and the `pROC` R package.

**Effect size warning.** With ~26K test users and millions of (user, product) rows, even tiny metric differences can be "statistically significant" while being practically negligible. We report effect sizes (actual metric differences) alongside p-values and treat F1 differences below ~0.005 as practically equivalent regardless of significance.

---

## 7. Compute Environment: NYU HPC OOD

The full Instacart dataset has ~10M (user, product) rows after feature engineering, which approaches the memory limits of typical laptops and makes hyperparameter tuning slow. Following the practice of public implementations on this dataset (Olson-Swanson 2018; syntheticjohn 2019, both of whom used AWS for final fits), we use **NYU's Greene cluster via Open OnDemand (OOD)** for the compute-heavy phases of the project.

### 7.1 Why HPC for this project

Concrete pain points HPC solves:

- **Memory headroom.** Feature engineering on ~10M rows in pandas peaks at ~10 GB during merges. Yu (2017) explicitly documented that 8 GB local RAM made tuning painful; we avoid this by using HPC nodes with 32–64+ GB.
- **Parallel tuning runs.** All four models can be tuned simultaneously via separate Slurm jobs rather than sequentially on one machine.
- **Untethered long jobs.** XGBoost randomized search with 30 samples × 5 folds × early stopping takes 1–2 hours; submitting via Slurm avoids tying up a laptop.
- **Reproducibility.** Slurm scripts with explicit resource requests are more reproducible than "ran on my laptop."

### 7.2 Workflow split: local vs. HPC

We follow the standard split used by Olson-Swanson and syntheticjohn:

| Phase | Where | Why |
|---|---|---|
| EDA (Phase 1) | Local laptop | Fast iteration on plots and summaries; small samples sufficient |
| Feature pipeline development | Local laptop on 10% subsample | Debugging is faster locally; no Slurm queue wait |
| Feature pipeline (full data) | HPC OOD | Memory-bound on full data |
| Hyperparameter tuning (Phase 4) | HPC OOD | Long jobs, parallelizable across models |
| Final model refit on full training data | HPC OOD | Memory-bound |
| Error analysis (Phase 5) | Local laptop | Working with predictions only (~MB-scale data) |
| Report writing | Local laptop | Standard work |

### 7.3 NYU Greene specifics

**Access.** NYU Greene is accessed via Open OnDemand at `https://ood.hpc.nyu.edu`. OOD provides a browser-based Jupyter Lab interface that removes most of the SSH/tmux friction of traditional HPC workflows. We will use the Jupyter app for interactive development and `sbatch` for long unattended runs.

**Partitions and resource requests.**

| Job type | Partition | Resources requested |
|---|---|---|
| Interactive development (OOD Jupyter) | `interactive` or `short` | 1 node, 8 CPUs, 32 GB RAM, 4 hours |
| Feature engineering on full data | `cpu` | 1 node, 8 CPUs, 64 GB RAM, 4 hours |
| Logistic regression tuning | `short` | 1 node, 4 CPUs, 16 GB RAM, 2 hours |
| Random Forest / AdaBoost tuning | `short` | 1 node, 8 CPUs, 32 GB RAM, 4 hours |
| XGBoost tuning | `short` | 1 node, 8 CPUs, 32 GB RAM, 4 hours |
| Final model refits | `cpu` | 1 node, 8 CPUs, 64 GB RAM, 6 hours |

We do **not** request GPU resources — none of our four models benefit from GPU on tabular data of this size. (XGBoost has `tree_method='gpu_hist'` but the speedup at 10M rows is modest and not worth the more constrained GPU queue.)

**Storage layout.**

```
/scratch/<netid>/instacart/
├── data/
│   ├── raw/              # Original Kaggle CSVs (~700 MB)
│   └── processed/        # Engineered features (parquet, ~2 GB)
├── src/
│   ├── eda.ipynb
│   ├── features.py
│   ├── tune_logreg.py
│   ├── tune_rf.py
│   ├── tune_adaboost.py
│   ├── tune_xgboost.py
│   └── final_fit.py
├── slurm/
│   └── *.sbatch          # Job submission scripts
├── models/
│   └── *.pkl             # Trained model artifacts
└── results/
    └── *.csv             # Tuning logs, predictions, metrics
```

`/scratch` has high quotas (~5 TB) but is purged after 60 days of inactivity. Final results and code are also pushed to a Git repo for permanent storage.

### 7.4 Slurm submission pattern

Each tuning job is wrapped in an `sbatch` script. The four model-tuning jobs are submitted in parallel and run on separate nodes simultaneously, cutting wall-clock time roughly 4× vs. sequential local execution.

Example skeleton (`tune_xgboost.sbatch`):

```bash
#!/bin/bash
#SBATCH --job-name=xgb_tune
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/xgb_tune_%j.out

module purge
module load anaconda3/2024.06
source activate instacart

python src/tune_xgboost.py \
    --data /scratch/$USER/instacart/data/processed/train.parquet \
    --n_iter 30 \
    --n_jobs 8 \
    --output /scratch/$USER/instacart/results/xgb_tuning.csv
```

### 7.5 Environment management

We use a conda environment defined in `environment.yml` and built once on Greene:

```yaml
name: instacart
dependencies:
  - python=3.11
  - numpy, pandas, scikit-learn, xgboost
  - matplotlib, seaborn
  - jupyterlab, ipykernel
  - pyarrow  # for parquet
```

The environment is registered as a Jupyter kernel so OOD's Jupyter app picks it up automatically. The same `environment.yml` is used locally to keep dev/HPC parity.

### 7.6 Practical guardrails

- **Never run heavy code on the login node.** All real computation goes through Slurm or interactive OOD sessions. Login-node abuse is the single most common way to get account warnings.
- **Always test on a 10% subsample first.** Submit the full job only after the subsample run completes successfully end-to-end. This avoids wasting 4 hours of queue time on a typo.
- **Save intermediate artifacts.** Feature-engineered data is saved as `parquet`, model artifacts as `pickle`, tuning results as `csv` — so re-running a single phase doesn't require re-running everything upstream.
- **Set random seeds.** Every tuning run sets `random_state=42` (or equivalent) at the top of the script, both for reproducibility and so we can compare runs cleanly.

### 7.7 Fallback if HPC is unavailable

If NYU HPC is down or our allocation is exhausted, the project remains feasible on a 16 GB laptop using a 20% user subsample throughout (rather than full data for the final fit). This trades ~0.01–0.02 F1 points for portability — acceptable for a class project. We do not plan around this fallback but document it as a contingency.

---

## 8. Deliverables and Timeline

| Phase | Deliverable | Estimated effort |
|---|---|---|
| 1. EDA | Notebook with summary stats, distribution plots, subgroup definitions | 1 week |
| 2. Feature engineering | Feature pipeline (Python script), 36-feature dataset | 1.5 weeks |
| 3. Splitting & leakage checks | User-level split + leakage verification report | 0.5 weeks |
| 4. Model training | Tuned models (LR, RF, AdaBoost, XGBoost), CV results | 1.5 weeks |
| 5. Error analysis | Comparison report with confusion matrices, subgroup analyses, feature importance | 1 week |
| Final | Project report and presentation | 0.5 weeks |

**Total: ~6 weeks.**

---

## 9. Limitations and Future Work

**Excluded by design:**

- Neural network models (MLP, RNN, Transformer) — orthogonal to our classical-comparison framing
- Learned features (word2vec, NNMF, KNN-based) — adds a separate modeling pipeline that complicates comparison
- Per-order F1-threshold optimization (Onodera's technique) — improves leaderboard score but not model comparison
- Stacked ensembles (Vasquez's approach) — outside scope of single-model comparison

**Genuine limitations:**

- We don't observe the user's *true* next purchase intent, only the labeled final order in the dataset
- The Instacart data caps `days_since_prior_order` at 30, which loses information about long-lapsed users
- Our subgroup definitions in Phase 5 are pre-registered hypotheses; we don't claim post-hoc subgroup discoveries

---

## 10. References

- Belsley, D. A., Kuh, E. & Welsch, R. E. (1980). *Regression Diagnostics: Identifying Influential Data and Sources of Collinearity*. New York: Wiley.
- Bergstra, J. & Bengio, Y. (2012). "Random search for hyper-parameter optimization." *Journal of Machine Learning Research*, 13, 281–305.
- Bhat, A. (2022). "Machine learning case study: Instacart market basket analysis." Medium. https://anandbhat92.medium.com/machine-learning-case-study-instacart-market-basket-analysis-1ba548d644ca
- DeLong, E. R., DeLong, D. M. & Clarke-Pearson, D. L. (1988). "Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach." *Biometrics*, 44(3), 837–845.
- Micci-Barreca, D. (2001). "A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems." *ACM SIGKDD Explorations Newsletter*, 3(1), 27–32. https://doi.org/10.1145/507533.507538
- Olson-Swanson, A. (2018). "Project 3 — Predicting products from Instacart data." Medium. https://medium.com/@anders.olsonswanson/project-3-predicting-products-from-instacart-data-854e3a6dbc7e
- Onodera, K. (2017). "Instacart Market Basket Analysis: 2nd place solution." Kaggle Blog. https://medium.com/kaggle-blog/instacart-market-basket-analysis-feda2700cded
- Pan, Y. (2017). "Instacart solution." GitHub. https://github.com/panyao/instacart-solution
- Rich, A. (2017). "23rd place solution for Kaggle's Instacart Market Basket Analysis competition." GitHub. https://github.com/alexanderrich/instacart-analysis
- syntheticjohn (2019). "Instacart order classification." GitHub. https://github.com/syntheticjohn/instacart_order_classification
- Vasquez, S. (2017). "Instacart Market Basket Analysis: 3rd place solution." GitHub. https://github.com/sjvasquez/instacart-basket-prediction
- Yu, D. (2017). "Kaggle Instacart Market Basket Analysis." GitHub. https://github.com/yudong-94/Kaggle-Instacart-Market-Basket-Analysis
- Dataset: Kaggle, *Instacart Market Basket Analysis*. https://www.kaggle.com/c/instacart-market-basket-analysis
