"""
evaluate_liar.py
----------------
Cross-domain evaluation — tests the FakeNewsNet-trained model
on the LIAR dataset without any retraining.

This is the key novelty claim of the research paper:
the model generalizes across domains.

Run after train_model.py:
    python evaluate_liar.py
"""

import pandas as pd
import numpy as np
import joblib
import re
from sklearn.metrics import classification_report, accuracy_score

# ── Load trained model ────────────────────────────────────────────────
print("🔄  Loading model trained on FakeNewsNet...")
pipeline = joblib.load("pipeline.pkl")
print("✅  Model loaded.\n")


# ── LIAR label mapping ────────────────────────────────────────────────
# LIAR has 6 labels — map to binary Fake / Real
FAKE_LABELS = ["pants-fire", "false", "barely-true"]
REAL_LABELS = ["half-true", "mostly-true", "true"]


# ── Text cleaning ─────────────────────────────────────────────────────
def clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Load LIAR test set ────────────────────────────────────────────────
print("📂  Loading LIAR dataset...")

# LIAR TSV columns (no header in file)
columns = [
    "id", "label", "statement", "subject",
    "speaker", "job", "state", "party",
    "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_fire_count", "context"
]

try:
    test_df  = pd.read_csv("test.tsv",  sep="\t", names=columns, on_bad_lines="skip")
    valid_df = pd.read_csv("valid.tsv", sep="\t", names=columns, on_bad_lines="skip")
    liar_df  = pd.concat([test_df, valid_df], ignore_index=True)
except FileNotFoundError:
    print("❌  test.tsv / valid.tsv not found.")
    print("    Download from: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip")
    print("    Place test.tsv and valid.tsv in the backend/ folder.")
    exit(1)

# Keep only binary-mappable labels
liar_df = liar_df[liar_df["label"].isin(FAKE_LABELS + REAL_LABELS)].copy()

# Map to binary
liar_df["binary_label"] = liar_df["label"].apply(
    lambda x: "Fake" if x in FAKE_LABELS else "Real"
)

print(f"    Total LIAR samples: {len(liar_df)}")
print(f"    Fake: {(liar_df['binary_label']=='Fake').sum()}")
print(f"    Real: {(liar_df['binary_label']=='Real').sum()}\n")


# ── Prepare features ──────────────────────────────────────────────────
liar_df["content"] = liar_df["statement"].apply(clean)
liar_df = liar_df[liar_df["content"].str.len() > 5]

X_liar = liar_df["content"].values
y_liar = (liar_df["binary_label"] == "Fake").astype(int).values


# ── Predict — NO retraining, direct inference ─────────────────────────
print("🔍  Running cross-domain evaluation on LIAR...")
print("    (Model was trained on FakeNewsNet — NOT on LIAR)\n")

y_pred = pipeline.predict(X_liar)
acc    = accuracy_score(y_liar, y_pred)

print(f"✅  Cross-domain accuracy on LIAR: {acc:.4f}  ({acc*100:.2f}%)\n")
print(classification_report(y_liar, y_pred, target_names=["Real", "Fake"]))


# ── Per original label breakdown ──────────────────────────────────────
print("\n📊  Breakdown by original LIAR label:")
for orig_label in FAKE_LABELS + REAL_LABELS:
    subset = liar_df[liar_df["label"] == orig_label]
    if len(subset) == 0:
        continue
    X_sub = subset["content"].values
    y_sub = (subset["binary_label"] == "Fake").astype(int).values
    y_sub_pred = pipeline.predict(X_sub)
    sub_acc = accuracy_score(y_sub, y_sub_pred)
    mapped = "→ Fake" if orig_label in FAKE_LABELS else "→ Real"
    print(f"    {orig_label:<20} {mapped}  |  accuracy: {sub_acc:.4f}  ({len(subset)} samples)")


# ── Summary for paper ─────────────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS FOR RESEARCH PAPER")
print("="*60)
print(f"  Training domain  : FakeNewsNet (PolitiFact + GossipCop)")
print(f"  Testing domain   : LIAR Dataset (cross-domain)")
print(f"  Cross-domain acc : {acc*100:.2f}%")
print(f"  Samples tested   : {len(liar_df)}")
print("="*60)