import pandas as pd
import joblib
import re
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

print("📂 Loading dataset...")
df = pd.read_csv("news_articles.csv", encoding="utf-8", on_bad_lines="skip")

# Keep valid labels
df = df[df["label"].isin(["Fake", "Real"])].copy()

# ── CLEAN FUNCTION ──
def clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Fill missing
df["title"] = df["title"].fillna("")
df["text"]  = df["text"].fillna("")

# ── KEY FIX: use title + partial text ──
df["content"] = (df["title"] + " " + df["text"].str[:300]).apply(clean)

# Remove empty
df = df[df["content"].str.len() > 20]

# Labels
df["label"] = (df["label"] == "Fake").astype(int)

X = df["content"].values
y = df["label"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── MODEL ──
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2
    )),
    ("clf", LogisticRegression(
        max_iter=1000
    ))
])

print("🏋️ Training...")
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {acc:.4f}\n")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

# Save
joblib.dump(pipeline, "pipeline.pkl")

print("💾 Saved pipeline.pkl")
print("🎉 Done")