import pandas as pd
import numpy as np
import joblib
import re
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

print("📂 Loading dataset...")
df = pd.read_csv("news_articles.csv", encoding="utf-8", on_bad_lines="skip")

df = df[df["label"].isin(["Fake", "Real"])].copy()

# ── CLEAN ──
def clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["title"] = df["title"].fillna("")
df["text"]  = df["text"].fillna("")

# ── KEY CHANGE: USE TITLE ONLY ──
df["content"] = df["title"].apply(clean)

# Remove short junk
df = df[df["content"].str.len() > 10]

# ── ADD SYNTHETIC CLAIM DATA ──
fake_samples = [
    "government secretly poisoning citizens",
    "officials hiding toxic chemicals in food",
    "secret experiments conducted on public",
    "mass surveillance injecting substances",
    "hidden agenda to control population"
]

real_samples = [
    "government releases annual budget report",
    "officials announce new healthcare policy",
    "study shows economic growth increased",
    "scientists publish research findings",
    "new law passed in parliament today"
]

extra_df = pd.DataFrame({
    "content": fake_samples + real_samples,
    "label": [1]*len(fake_samples) + [0]*len(real_samples)
})

# ── LABELS ──
df["label"] = (df["label"] == "Fake").astype(int)

# Merge
df = pd.concat([df[["content", "label"]], extra_df]).reset_index(drop=True)

X = df["content"].values
y = df["label"].values

# ── SPLIT ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── MODEL ──
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2
    )),
    ("clf", LogisticRegression(
        C=3.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

print("🏋️ Training...")
pipeline.fit(X_train, y_train)

# ── EVAL ──
y_pred = pipeline.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

# ── SAVE ──
joblib.dump(pipeline, "pipeline.pkl")

print("💾 Saved pipeline.pkl")
print("🎉 Done.")