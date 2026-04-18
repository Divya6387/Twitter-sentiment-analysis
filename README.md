# Twitter Sentiment Analysis — Hate Speech Detection

Binary text classification to detect racist or sexist content in tweets.

Analytics Vidhya Hackathon — codefest_linguipedia

Evaluation Metric — F1-Score

---

## Problem Statement

The goal is to detect hate speech in tweets. A tweet is labeled 1 if it contains
racist or sexist sentiment, and 0 otherwise. Given labeled training tweets,
the task is to predict labels on unseen test tweets.

- Train set — 31,962 tweets (labeled)
- Test set  — 17,197 tweets (unlabeled, to predict)
- Severe class imbalance — 93% normal (0) vs 7% hate speech (1)

## Setup and Usage

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Add data files

Place train_E6oV3lV.csv and test_tweets_anuFYb8.csv inside the data/ folder.

Update TRAIN_PATH and TEST_PATH at the top of solution.py if needed.

### Step 4 — Run the pipeline

```bash
python solution.py
```

The script runs end-to-end and saves all outputs automatically.

---

## Data Dictionary

| Column | Description |
|---|---|
| id | Unique tweet ID |
| label | 0 = Normal tweet, 1 = Racist or Sexist tweet |
| tweet | Raw tweet text |

---

## Step-by-Step Pipeline Explanation

### Step 1 — Load Data

Reads train and test CSVs using pandas. Prints label distribution and class imbalance ratio.
The data has a severe 13:1 imbalance which requires special handling.

### Step 2 — Exploratory Data Analysis

Three EDA plots are generated and saved:

**eda_overview.png** — Three subplots side by side:
- Label distribution bar chart showing 29,720 normal vs 2,242 hate speech tweets
- Tweet length histogram comparing word count distribution for both classes
- Top-10 most frequent hashtags found in hate speech tweets

**wordcloud_normal.png** — A green word cloud of the most common words in normal tweets. Larger words appear more frequently. Helps visualise vocabulary used in everyday tweets.

**wordcloud_hate.png** — A red word cloud for hate speech tweets. Reveals patterns and specific language associated with racist or sexist content.

### Step 3 — Text Preprocessing

Each tweet is cleaned with the following steps applied in order:

1. Remove @mentions — usernames add no semantic value
2. Remove URLs — links do not help classify sentiment
3. Lowercase everything — standardises vocabulary
4. Remove all non-letter characters — strips punctuation, numbers, emojis
5. Tokenise on whitespace
6. Remove stopwords — common words like "the", "is", "and" are filtered out
7. Remove very short tokens — tokens of length 2 or less are dropped

A hand-coded stopword list is used so no external NLTK download is required.

### Step 4 — TF-IDF Feature Extraction

TF-IDF (Term Frequency — Inverse Document Frequency) converts cleaned text into
numerical feature vectors.

Settings used:
- ngram_range (1, 2) — includes both single words and two-word phrases (bigrams)
- max_features 50,000 — keeps the 50,000 most informative tokens
- min_df 3 — ignores tokens appearing in fewer than 3 tweets (noise reduction)
- max_df 0.90 — ignores tokens appearing in more than 90% of tweets (too common)
- sublinear_tf True — applies log(1 + tf) to reduce the impact of very frequent terms

Result: a sparse matrix of shape (31962, ~16000)

### Step 5 — Model Training with 5-Fold Stratified Cross-Validation

Two models are trained and compared:

**Logistic Regression**
- Simple, fast, and interpretable linear classifier
- C=5 controls regularisation strength
- class_weight=balanced automatically up-weights the minority class
- CV F1 around 0.68

**Linear SVC (Calibrated)**
- Support Vector Classifier with a linear kernel — often top performer on text data
- Wrapped in CalibratedClassifierCV to output probability scores
- class_weight=balanced applied here too
- CV F1 around 0.69

Stratified K-Fold ensures each fold has the same 13:1 class ratio as the full dataset.

**cv_comparison.png** — Bar chart comparing F1 scores of both models with error bars
showing the standard deviation across 5 folds.

### Step 6 — Fit Final Models

Both models are retrained on the full training set (all 31,962 tweets) after
cross-validation selects the configuration.

### Step 7 — Evaluation

**confusion_matrices.png** — A 2x2 heatmap for each model showing:
- True Positives — correctly predicted hate speech
- True Negatives — correctly predicted normal tweets
- False Positives — normal tweets wrongly flagged as hate speech
- False Negatives — hate speech tweets missed by the model

The confusion matrix reveals the trade-off between precision and recall.
For hate speech detection, recall (catching actual hate speech) matters more
than precision (avoiding false alarms).

**classification_report** — Printed to console. Shows precision, recall, and F1
for each class individually.

**roc_curves.png** — ROC curve for each model. The x-axis is the False Positive Rate
and the y-axis is the True Positive Rate. AUC (Area Under Curve) closer to 1.0
means better discrimination ability. Both models score above 0.95 AUC.

**top_features.png** — Two horizontal bar charts:
- Left: top-20 tokens with the highest positive LR coefficients — these push
  the model toward predicting hate speech
- Right: top-20 tokens with the most negative coefficients — these push the
  model toward predicting normal tweets

This makes the model interpretable and shows which words are most predictive.

### Step 8 — Weighted Ensemble and Threshold Tuning

**Ensemble** — Predictions from both models are combined using a weighted average.
Each model's weight equals its CV F1 score so the better model contributes more.

**Threshold tuning** — By default classifiers use 0.5 as the decision boundary.
Because the dataset is imbalanced, a lower threshold increases recall for the
minority (hate speech) class. A grid search over thresholds from 0.10 to 0.90
finds the value that maximises F1 on the training set.

**threshold_vs_f1.png** — Line chart showing F1 vs threshold. The optimal threshold
is marked with a red dashed line. This is applied to the test set.

### Step 9 — Submission File

test_predictions.csv is saved to outputs/ with one label per row (0 or 1),
in the same order as the test tweets file.

---

## Model Results Summary

| Model | CV F1 Score |
|---|---|
| Logistic Regression | 0.6798 |
| Linear SVC (Calibrated) | 0.6876 |
| Weighted Ensemble | Best on test set |

---

## Why These Choices Were Made

**Why TF-IDF and not deep learning?**
TF-IDF with linear models is a strong, fast, and interpretable baseline for
short text classification. It requires no GPU and trains in seconds. Deep
learning models like BERT would likely improve F1 but are significantly
more complex to set up and run.

**Why class_weight=balanced?**
With a 13:1 imbalance, a model that always predicts Normal would get 93%
accuracy but 0 F1 on hate speech. Setting class_weight=balanced tells the
model to treat each hate speech error as 13x more costly than a normal error.

**Why F1 and not accuracy?**
F1 is the harmonic mean of precision and recall. On imbalanced data, accuracy
is misleading. F1 specifically measures performance on the minority class which
is exactly what this problem cares about.

**Why a calibration wrapper on SVC?**
LinearSVC does not natively produce probability scores. CalibratedClassifierCV
adds a Platt scaling layer so predict_proba works, enabling soft ensemble voting
and threshold tuning.

---

## Key Insights from EDA

- Hate speech tweets are on average slightly longer than normal tweets
- Specific hashtags strongly correlate with hateful content
- The vocabulary of hate speech is concentrated around identity-based language
- Normal tweets have a much more diverse and general vocabulary

---

## Submission Format

The file test_predictions.csv should contain one column named label with one
prediction per row (0 or 1), matching the order of tweets in the test file.

```
label
0
1
0
0
...
```

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
wordcloud
```

No NLTK or internet connection required to run the pipeline.
