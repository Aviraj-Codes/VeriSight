import pandas as pd
import numpy as np
import joblib
import re
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# ── 1. Load dataset ────
print("📂  Loading dataset...")
df = pd.read_csv("news_articles.csv", encoding="utf-8", on_bad_lines="skip")

# Keep only rows with a valid label
df = df[df["label"].isin(["Fake", "Real"])].copy()
print(f"    Rows after filtering: {len(df)}  (Fake: {(df['label']=='Fake').sum()}, Real: {(df['label']=='Real').sum()})")

# ── 2. Balance the dataset (undersample Fake to match Real count) ─────
real_df = df[df["label"] == "Real"]
fake_df = df[df["label"] == "Fake"].sample(n=len(real_df), random_state=42)
df = pd.concat([real_df, fake_df]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"    After balancing — Fake: {(df['label']=='Fake').sum()}, Real: {(df['label']=='Real').sum()}")

# ── 3. Build a combined text feature ────
def clean(text):
    """Lowercase, strip URLs, punctuation, and extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)                # keep letters only
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Combine title + body for richer signal
df["title"]   = df["title"].fillna("")
df["text"]    = df["text"].fillna("")
df["content"] = df["title"] + " " + df["text"]
df["content"] = df["content"].apply(clean)

# Drop empty rows
df = df[df["content"].str.len() > 20]
X = df["content"].values
y = (df["label"] == "Fake").astype(int).values   # 1 = Fake, 0 = Real

# ── 4. Train / test split ────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n🔀  Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── 5. Build pipeline (TF-IDF + Logistic Regression) ────
print("\n🏋️   Training model...")
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=30_000,
        ngram_range=(1, 2),     # unigrams + bigrams
        sublinear_tf=True,      # apply log(1+tf)
        min_df=2,
        strip_accents="unicode",
        analyzer="word",
    )),
    ("clf", LogisticRegression(
        C=5.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight=None,      # balanced dataset — no need for class weighting
        random_state=42,
    )),
])
pipeline.fit(X_train, y_train)

# ── 6. Evaluate ───
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅  Test accuracy: {acc:.4f}  ({acc*100:.2f}%)\n")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

# ── 7. Save artefacts ───
vectorizer = pipeline.named_steps["tfidf"]
model      = pipeline.named_steps["clf"]
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(model,      "model.pkl")
joblib.dump(pipeline,   "pipeline.pkl")
print("💾  Saved: vectorizer.pkl, model.pkl, pipeline.pkl")
print("\n🎉  Done! You can now start the Flask server with:  python app.py")