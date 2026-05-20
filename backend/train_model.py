"""
train_model.py
--------------
Trains a fake-news classifier on FakeNewsNet datasets:
  - politifact_real.csv
  - politifact_fake.csv
  - gossipcop_real.csv
  - gossipcop_fake.csv

Also supports the original news_articles.csv for cross-validation.

Saves:
  - pipeline.pkl    (full pipeline — used by app.py)
  - vectorizer.pkl  (TF-IDF)
  - model.pkl       (Logistic Regression)

Run once before starting the Flask server:
    python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import re
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Fix CSV field size limit for large files
csv.field_size_limit(10**7)


# ── 1. Load FakeNewsNet datasets ──────────────────────────────────────
print("📂  Loading FakeNewsNet datasets...")

def load_fakenewsnet(real_path, fake_path, source_name):
    """Load a real + fake CSV pair and return a combined DataFrame."""
    real_df = pd.read_csv(real_path, encoding="utf-8", on_bad_lines="skip")
    fake_df = pd.read_csv(fake_path, encoding="utf-8", on_bad_lines="skip")
    real_df["label"] = "Real"
    fake_df["label"] = "Fake"
    combined = pd.concat([real_df, fake_df], ignore_index=True)
    combined["source"] = source_name
    print(f"    {source_name}: {len(real_df)} Real + {len(fake_df)} Fake = {len(combined)} total")
    return combined

politifact = load_fakenewsnet(
    "politifact_real.csv",
    "politifact_fake.csv",
    "PolitiFact"
)

gossipcop = load_fakenewsnet(
    "gossipcop_real.csv",
    "gossipcop_fake.csv",
    "GossipCop"
)

# ── 2. Also load original dataset if available ────────────────────────
extra_df = None
try:
    extra = pd.read_csv("news_articles.csv", encoding="utf-8", on_bad_lines="skip")
    extra = extra[extra["label"].isin(["Fake", "Real"])].copy()
    # Combine title + text columns
    extra["title"] = extra.get("title", pd.Series(dtype=str)).fillna("")
    extra["text"]  = extra.get("text",  pd.Series(dtype=str)).fillna("")
    extra["content"] = extra["title"] + " " + extra["text"]
    extra["source"] = "Original"
    extra_df = extra[["content", "label", "source"]]
    print(f"    Original CSV: {len(extra_df)} rows loaded")
except FileNotFoundError:
    print("    Original news_articles.csv not found — skipping")


# ── 3. Combine all data ───────────────────────────────────────────────
print("\n🔗  Combining datasets...")

# FakeNewsNet only has title column — use it as content
politifact["content"] = politifact["title"].fillna("")
gossipcop["content"]  = gossipcop["title"].fillna("")

frames = [
    politifact[["content", "label", "source"]],
    gossipcop[["content", "label", "source"]],
]
if extra_df is not None:
    frames.append(extra_df)

df = pd.concat(frames, ignore_index=True)

print(f"    Total rows: {len(df)}")
print(f"    Fake: {(df['label'] == 'Fake').sum()}")
print(f"    Real: {(df['label'] == 'Real').sum()}")


# ── 4. Balance the dataset ────────────────────────────────────────────
print("\n⚖️   Balancing dataset...")
real_df = df[df["label"] == "Real"]
fake_df = df[df["label"] == "Fake"]

# Undersample the majority class
min_count = min(len(real_df), len(fake_df))
real_df = real_df.sample(n=min_count, random_state=42)
fake_df = fake_df.sample(n=min_count, random_state=42)

df = pd.concat([real_df, fake_df]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"    After balancing — Fake: {(df['label']=='Fake').sum()}, Real: {(df['label']=='Real').sum()}")


# ── 5. Clean text ─────────────────────────────────────────────────────
def clean(text):
    """Lowercase, strip URLs, punctuation, and extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["content"] = df["content"].apply(clean)

# Drop rows with very short content
df = df[df["content"].str.len() > 10].reset_index(drop=True)

X = df["content"].values
y = (df["label"] == "Fake").astype(int).values   # 1 = Fake, 0 = Real


# ── 6. Train / test split ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n🔀  Train: {len(X_train)}  |  Test: {len(X_test)}")


# ── 7. Build pipeline ─────────────────────────────────────────────────
print("\n🏋️   Training model...")
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=50_000,       # larger vocab for bigger dataset
        ngram_range=(1, 3),        # unigrams + bigrams + trigrams
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
        analyzer="word",
    )),
    ("clf", LogisticRegression(
        C=3.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight=None,         # dataset is already balanced
        random_state=42,
    )),
])

pipeline.fit(X_train, y_train)


# ── 8. Evaluate ───────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅  Test accuracy: {acc:.4f}  ({acc*100:.2f}%)\n")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))


# ── 9. Per-source breakdown ───────────────────────────────────────────
print("\n📊  Per-source accuracy:")
for source in df["source"].unique():
    src_mask = df["source"] == source
    src_X = df.loc[src_mask, "content"].values
    src_y = (df.loc[src_mask, "label"] == "Fake").astype(int).values
    if len(src_X) > 0:
        src_pred = pipeline.predict(src_X)
        src_acc  = accuracy_score(src_y, src_pred)
        print(f"    {source}: {src_acc:.4f} ({src_acc*100:.2f}%) — {len(src_X)} samples")


# ── 10. Save artefacts ────────────────────────────────────────────────
vectorizer = pipeline.named_steps["tfidf"]
model      = pipeline.named_steps["clf"]

joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(model,      "model.pkl")
joblib.dump(pipeline,   "pipeline.pkl")

print("\n💾  Saved: vectorizer.pkl, model.pkl, pipeline.pkl")
print("\n🎉  Done! You can now start the Flask server with:  python app.py")