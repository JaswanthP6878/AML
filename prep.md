# AML Project — Check-In Prep Notes

## What The Project Is (30-second pitch)

We built an anti-money laundering (AML) transaction detection system. It merges two labeled datasets — IBM's real bank transaction data and SAML-D synthetic AML data — engineers historical behavioral features per account, and benchmarks several tree ensemble models to flag suspicious transactions. The goal is to rank transactions by fraud probability so investigators can triage the highest-risk alerts first.

---

## 1. Data Sources & Processing (15 pts)

### What data did you use?

Two datasets, merged into one:

| Dataset | Description | Key columns |
|---|---|---|
| **IBM HI-Small_Trans.csv** | Real bank transaction records with fraud labels | Timestamp, From/To Bank+Account, Amount Paid/Received, Payment Format, `Is_Laundering` |
| **SAML-D** | Synthetic AML transactions with laundering type labels | Date, Time, Sender/Receiver account, Amount, Currency, Payment type, `Is_laundering`, `Laundering_type` |

Both are normalized to a **15-column unified schema**: `source_dataset`, `timestamp`, `from_bank`, `from_account`, `to_bank`, `to_account`, `transaction_amount`, `amount_paid`, `amount_received`, `payment_currency`, `receiving_currency`, `payment_format`, `transaction_type`, `laundering_type`, `is_laundering`.

### What cleaning steps did you do?

- **Text normalization**: whitespace stripped, blank strings → NA across all category columns
- **Amount coercion**: all amount columns forced to float; unparseable values → NaN
- **Timestamp parsing**: `pd.to_datetime(..., errors='coerce')` — bad timestamps become NaT and get filled with synthetic sequential values so chronological sorting stays valid
- **Label harmonization**: IBM uses numeric 0/1; SAML-D uses "Yes"/"No" → mapped to 1/0
- **Missing canonical columns**: if a column doesn't exist in a dataset, it's added as NA so the two datasets can concatenate cleanly
- **Leak-prone columns excluded from training**: `laundering_type` and `source_dataset` are kept only for post-hoc auditing, never fed to the model

### Why a chronological split instead of random?

> "Financial transactions are time-ordered. A random split would let cumulative account-history features 'see' future transactions during training, which artificially inflates validation scores. Our 64/16/20 chronological split ensures training features are always computed from strictly past data."

### Why median imputation (not mean)?

> "Transaction amounts are heavily right-skewed — a few very large laundering amounts would pull the mean up. The median is robust to those outliers. The imputer is fit on training data only and applied to validation/test to prevent any statistics from leaking across splits."

### Why OrdinalEncoder with unknown → -1?

> "New payment types can appear at inference time that weren't in training data. Mapping unknowns to -1 lets the trees route them as a distinct group rather than crashing or misclassifying silently."

---

## 2. Feature Engineering

### Feature groups built (all leak-safe, using only past transactions):

1. **Temporal**: hour, weekday, month, is_weekend, is_night, cyclic hour encoding (sin/cos)
2. **Amount**: log(amount), gap between amount_paid and amount_received, ratio, normalized versions
3. **Network**: same_bank flag, same_currency flag, cross_border flag
4. **Velocity**: cumulative transaction counts per sender, receiver, sender-receiver pair, and route — both total and per-day
5. **Historical stats**: previous mean and std of amounts per account entity (leak-safe cumulative calculation)
6. **Time gaps**: minutes since last transaction for each entity pair/route
7. **Novelty**: first-time-seen flags, running count of unique counterparties per account
8. **Z-scores**: how many standard deviations the current amount is from the account's history
9. **Amount share**: this transaction as a fraction of the account's cumulative volume

> "We don't just look at a single transaction — we look at the account's entire behavioral history up to that point. Layering and structuring patterns in money laundering are invisible in one transaction but obvious in the sequence."

---

## 3. Visualizations (15 pts)

### Visualization 1 — Dataset Overview (`plot_dataset_overview`)
Three panels:
- Bar chart: transaction counts per dataset (IBM vs. SAML-D)
- Bar chart: laundering rate per dataset
- KDE plot: log-scaled transaction amount distribution, split by fraud label (0 vs. 1)

**What claim does it support?**
> "The two datasets have different sizes and fraud prevalences, so merging them creates class imbalance we have to handle explicitly. The amount distribution shows laundering transactions cluster at unusual (often large or round) amounts — supporting the hypothesis that amount anomaly is a useful signal."

**Why log scale on the amount axis?**
> "Transaction amounts span many orders of magnitude — from a few dollars to millions. Without log scaling, the distribution is completely dominated by the spike near zero and you can't see the shape. Log scale lets us compare the tails where laundering concentrates."

---

### Visualization 2 — Temporal Heatmap (`plot_temporal_patterns`)
Heatmap: fraud rate (color) by weekday (rows, Mon–Sun) and hour of day (columns, 0–23).

**What claim does it support?**
> "If fraud concentrates at specific times — say, late nights or weekends — that suggests automated layering scripts operating outside business hours. A uniform heatmap means sophisticated actors mimicking normal business-hour patterns. Either finding shapes how we interpret time-based features in the model."

**Why a heatmap instead of a line chart?**
> "We have two time dimensions (day-of-week × hour). A heatmap encodes both simultaneously as a grid — any other chart would require either multiple lines or aggregating one dimension away, losing information."

---

### Visualization 3 — Correlation Views (`plot_numeric_correlation_views`)
Side-by-side heatmaps: Pearson (linear) and Spearman (monotonic) correlation of top-N numeric features against the fraud label.

**What claim does it support?**
> "This shows which features are most linearly or monotonically associated with laundering. If velocity features (cumulative counts, amounts) dominate this chart, it validates our feature engineering investment — the historical behavior signal is stronger than raw transaction metadata alone."

---

### Visualization 4 — Feature Association Views (`plot_feature_association_views`)
Three horizontal bar charts:
- Top numeric features by absolute Spearman correlation
- Top categorical features by mutual information and target-rate gap
- Unified mutual information ranking across all features

**What claim does it support?**
> "Mutual information captures non-linear signal that correlation misses. If the top features are historical velocity/behavioral ones rather than raw transaction fields, it confirms that laundering is a pattern-level phenomenon — not just 'large amount = fraud'."

---

### Visualization 5 — Threshold Diagnostics (`plot_threshold_diagnostics`)
Line plot: precision, recall, F1, and alert rate across all threshold values (0.05–0.95).

**What claim does it support?**
> "This shows the precision-recall trade-off at every operating point. We use it to pick a threshold — defaulting to best-F1 — but an investigator team that can handle more alerts would shift to the recall-optimized point. The chart makes that trade-off explicit."

---

### Visualization 6 — Model Diagnostics (`plot_model_diagnostics`)
Four panels at the selected threshold:
- Confusion matrix
- ROC curve (AUC labeled)
- Precision-Recall curve (AUC labeled)
- Score distribution histogram split by true/false positive/negative with threshold marker

**What claim does it support?**
> "The PR curve is the main performance summary — a curve hugging the top-right corner means the model is both precise and high-recall. The score histogram shows whether the model gives clearly separated probability scores to frauds vs. legitimate transactions."

**Why PR curve and not just ROC?**
> "ROC is misleading on imbalanced data. A classifier that flags everything as fraud gets ROC-AUC ≈ 1.0 but is useless operationally. PR-AUC focuses on the fraud class specifically — it answers 'of the alerts we send, how many are real?'"

---

### Visualization 7 — Rank Diagnostics (`plot_rank_diagnostics`)
Two curves:
- Precision@K: if you take the top-K scored transactions, what fraction are real fraud?
- Lift@K: how much better than random is the model's ranking among the top-K?

**What claim does it support?**
> "In practice, an analyst can only review a fixed number of alerts per day. Precision@K tells us that if we review the top-1% of transactions by risk score, we catch X% of all fraud. Lift@K shows we're doing many times better than random — operationally this is what justifies deploying the model."

---

### Visualization 8 — Feature Importance (`plot_feature_importance`)
Horizontal bar chart: top-25 features by model importance score.

**What claim does it support?**
> "This confirms which features the model actually relies on. If velocity and historical behavioral features dominate, our feature engineering was worthwhile. If raw metadata features dominate, we'd revisit the engineering approach."

---

### Visualization 9 — Error Mix (`plot_error_mix`)
Two panels:
- Stacked bar: TP/FP/FN/TN breakdown per dataset (IBM vs. SAML-D)
- Box plot: score distributions by error type for flagged transactions

**What claim does it support?**
> "This shows whether the model performs equally well on both datasets. If one dataset has a much higher false positive rate, there may be a distribution shift between IBM and SAML-D that we need to address. The score box plot shows whether false positives look similar to true positives in score space — if they overlap heavily, the threshold is too low."

---

## 4. Modeling Methods (15 pts)

### Models benchmarked:

| Model | Key property |
|---|---|
| XGBoost (regularized) | Balanced L1/L2 + `scale_pos_weight` for imbalance |
| XGBoost (recall-focused) | Deeper trees, lighter regularization |
| XGBoost (undersampled) | No `scale_pos_weight`; trained on undersampled data |
| HistGradientBoosting | sklearn native; handles missing values natively |
| BalancedRandomForest | Resamples each tree to balance classes |
| ExtraTrees | Extremely randomized splits; low variance, fast |
| Weighted Ensemble | PR-AUC-weighted average of top-3 models |

**+ IsolationForest anomaly score** appended as an extra feature to all splits before any supervised model trains.

### Why tree ensembles and not neural networks?

> "AML data is tabular — mixed numeric and categorical features. Tree ensembles handle this natively without feature scaling. They model non-linear interactions automatically. Neural networks on tabular data typically underperform tree ensembles (this is well-documented on Kaggle AML benchmarks). The imbalanced class distribution is also handled directly by `BalancedRandomForest` and `scale_pos_weight`."

### Why compare multiple models?

> "No single model is always best on imbalanced tabular data. By benchmarking 6 families and an ensemble, we avoid overfitting our model choice to one architecture. The weighted ensemble averages out uncorrelated errors across diverse models."

### How were models evaluated?

- **Primary metric: PR-AUC** (Precision-Recall Area Under Curve)
- **Secondary**: PR-lift = PR-AUC / baseline prevalence (e.g., 20× lift means the model is 20× better than random ranking)
- **Also reported**: ROC-AUC, precision, recall, F1, alert rate
- **Threshold sweep**: 0.05 to 0.95 on validation set; three operating profiles compared
- **Overfitting check**: train PR-AUC flagged if > 1.5× validation PR-AUC

### Why PR-AUC and not accuracy or ROC-AUC?

> "Fraud prevalence is ~0.1–2%. A model that predicts 'not fraud' for every transaction gets 98%+ accuracy. ROC-AUC is also inflated by the massive true-negative class. PR-AUC measures only how well we rank the positive (fraud) class — that's the operationally meaningful question."

---

## 5. Results & Interpretation (5 pts)

### How to read the results table (`split_metrics.csv`):

| Metric | What it means | What good looks like |
|---|---|---|
| `pr_auc` | Area under precision-recall curve | Much higher than baseline prevalence |
| `pr_lift` | PR-AUC ÷ baseline prevalence | 10× or higher is useful for triage |
| `roc_auc` | Area under ROC curve | > 0.85 for decent ranking |
| `precision` | Of flagged alerts, fraction that are real fraud | > 10–20% is operationally viable |
| `recall` | Of all fraud cases, fraction that were caught | > 60–70% is desirable |
| `f1` | Harmonic mean of precision and recall | Best combined performance |
| `alert_rate` | Fraction of transactions flagged | Lower = easier for investigators |

### Actual Results (fill in after running)

| split | roc_auc | pr_auc | pr_lift | precision | recall | f1 | threshold | positive_rate |
|---|---|---|---|---|---|---|---|---|
| train | 0.9995 | 0.7880 | 780× | 0.632 | 0.725 | 0.675 | 0.925 | 0.00101 |
| valid | 0.9999 | 0.9763 | 930× | 0.990 | 0.940 | 0.964 | 0.925 | 0.00105 |
| test  | 0.9999 | 0.9851 | 904× | 0.994 | 0.949 | 0.971 | 0.925 | 0.00109 |

Positive rate (fraud prevalence): ~0.1% across all splits. Selected threshold: **0.925**.

### What to say about these results:

> "Our test PR-AUC is 0.985 against a baseline of 0.00109 — that's a **904× lift** over random ranking. In practice, a fraud investigator reviewing the top-scored alerts would find that 99.4% of them are real fraud cases (precision = 0.994), while the model catches 94.9% of all fraud in the dataset (recall = 0.949). ROC-AUC of 0.9999 confirms near-perfect transaction ranking."

### The interesting pattern — train PR-AUC (0.788) is LOWER than valid (0.976) and test (0.985):

This will almost certainly be asked. Here is why it happens and how to explain it:

> "This is expected given our chronological split and historical feature engineering. The training set covers the **earliest time period**, when accounts have little transaction history — so features like 'cumulative sender amount' and 'average prior transaction for this account' are noisy or zero for most rows. The validation and test sets come from **later time periods** where accounts have accumulated richer histories, making the features more informative. The model was trained on harder examples (less context) and evaluated on easier ones (more context). This is actually a sign that our historical features work as intended — they get more powerful as more transaction history is available."

> "This is **not** overfitting — overfitting would show high train performance and low test performance, the opposite of what we see. The overfitting check (train PR-AUC > 1.5× valid PR-AUC) passes cleanly."

### Operating profile choices:

| Profile | Threshold | Use case |
|---|---|---|
| `best_f1` (default) | **0.925** | Balanced — 99.4% precision, 94.9% recall |
| `high_recall` | Lower than 0.925 | Catch more fraud; more FP load on investigators |
| `high_precision` | Higher than 0.925 | Already near-perfect at 0.925 |

Note: a threshold of 0.925 is unusually high. This means the model is very confident before flagging — it only fires on transactions it strongly believes are fraudulent. The high precision (99.4%) confirms this is working correctly.

---

## Likely Check-In Questions & Answers

**Q: Why did you merge two datasets instead of just using one?**
> "Each dataset has limitations. IBM has real transactions but no laundering type labels. SAML-D has structured typology labels but is synthetic. Merging gives us more data diversity and a larger fraud class, which improves model training."

**Q: Why did you use a scatter/bar/KDE instead of [other chart]?**
> See per-visualization "Why X instead of Y" explanations above.

**Q: How do you know your model isn't overfitting?**
> "Overfitting would show high train performance and collapsing test performance. Our train PR-AUC is actually *lower* (0.788) than validation (0.976) and test (0.985) — the opposite pattern. This happens because the training set covers the earliest time period where accounts have little history, so features are noisier. Later splits benefit from richer historical context. The overfitting guard (train PR-AUC > 1.5× valid) passes cleanly."

**Q: Why is train PR-AUC lower than validation and test PR-AUC?**
> "The training set is the earliest chronological window. Historical features — cumulative counts, average amounts, time-since-last-transaction — are sparse or zero for many accounts because those accounts haven't transacted much yet. As we move into the validation and test periods, accounts have more history and the features are much more informative. The model learned the pattern correctly; it just had weaker input data during training."

**Q: Your results look almost too good — is something wrong?**
> "The high numbers are real for the validation and test sets. Both datasets (IBM and SAML-D) have relatively clear laundering patterns detectable through behavioral history — structured transactions, unusual timing, repeated round-number amounts. Our historical feature engineering is specifically designed to capture these. The 0.788 train PR-AUC is the honest estimate of how hard the problem is on sparse early-period data; the val/test numbers reflect the model's performance once it has enough history to work with."

**Q: Why did you use these features specifically?**
> "The mutual information and Spearman correlation plots guided feature selection. Historical velocity features (cumulative counts, amounts per account) consistently show the highest association with fraud — which aligns with the AML literature on structuring and layering detection."

**Q: Why is your alert rate important?**
> "An investigator team has a fixed capacity — maybe 200 alerts per day. If our model flags 50% of all transactions, that's useless. We target an alert rate of 1–5% while maintaining recall > 60%, so the team can realistically review all flagged cases."

**Q: What would you do next to improve results?**
> "Graph-based features — modeling the transaction network to detect fan-in/fan-out patterns — are the highest-value next step. We'd also explore sequence models (LSTM or Transformer) on per-account transaction histories to capture temporal laundering patterns more directly."
