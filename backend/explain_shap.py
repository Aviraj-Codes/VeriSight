"""
explain_shap.py
---------------
Generates SHAP explanations for the VeriSight fake news classifier.
Shows which words pushed the prediction toward Fake or Real.

"""

import joblib
import shap
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots

# ── Load model ────────────────────────────────────────────────────────
print("🔄  Loading model...")
pipeline   = joblib.load("pipeline.pkl")
vectorizer = pipeline.named_steps["tfidf"]
model      = pipeline.named_steps["clf"]
print("✅  Model loaded.\n")

# ── Text cleaning ─────────────────────────────────────────────────────
def clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Test samples ──────────────────────────────────────────────────────
samples = {
    "fake_1": "Government puts microchips in vaccines to track and control citizens worldwide",
    "fake_2": "BREAKING: NASA confirms moon landing was staged in Hollywood studio by CIA",
    "fake_3": "Obama secret Muslim born Kenya deep state controls media and elections",
    "real_1": "Federal Reserve raises interest rates by 0.25 percent amid inflation concerns",
    "real_2": "Scientists discover new treatment for Alzheimers disease in clinical trials",
    "real_3": "United Nations calls for ceasefire as humanitarian crisis deepens in conflict zone",
}
cleaned_samples = {k: clean(v) for k, v in samples.items()}

# ── Build SHAP explainer ──────────────────────────────────────────────
print("🔍  Building SHAP explainer (this may take a minute)...")

# Use a background sample for the explainer
# LinearExplainer works well with TF-IDF + Logistic Regression
background_texts = list(cleaned_samples.values())
background_matrix = vectorizer.transform(background_texts)

explainer = shap.LinearExplainer(
    model,
    background_matrix,
    feature_perturbation="interventional"
)
print("✅  Explainer ready.\n")

# ── Generate SHAP values ──────────────────────────────────────────────
print("📊  Generating SHAP values...\n")

feature_names = np.array(vectorizer.get_feature_names_out())

def explain_text(label, text, cleaned):
    """Print top SHAP words for a single text."""
    vec   = vectorizer.transform([cleaned])
    shap_vals = explainer.shap_values(vec)

    # shap_vals shape: (1, n_features)
    # Positive = pushes toward Fake (class 1)
    # Negative = pushes toward Real (class 0)
    vals = shap_vals[0]

    # Get non-zero features
    nonzero = vec.nonzero()[1]
    if len(nonzero) == 0:
        print(f"  {label}: no features found\n")
        return

    nonzero_vals  = vals[nonzero]
    nonzero_names = feature_names[nonzero]

    # Sort by absolute SHAP value
    sorted_idx = np.argsort(np.abs(nonzero_vals))[::-1][:8]

    proba = pipeline.predict_proba([cleaned])[0]
    pred  = "Fake" if proba[1] >= 0.50 else "Real"

    print(f"  [{label}]")
    print(f"  Text      : {text[:80]}")
    print(f"  Prediction: {pred}  (Fake prob: {proba[1]:.4f})")
    print(f"  Top SHAP words:")
    for i in sorted_idx:
        direction = "→ Fake" if nonzero_vals[i] > 0 else "→ Real"
        print(f"    {nonzero_names[i]:<25} SHAP: {nonzero_vals[i]:+.4f}  {direction}")
    print()

for label, text in samples.items():
    explain_text(label, text, cleaned_samples[label])

# ── Save SHAP bar plot for paper ──────────────────────────────────────
print("📈  Saving SHAP plots for research paper...")

def save_shap_plot(label, text, cleaned, filename):
    """Save a SHAP bar chart for a single sample."""
    vec       = vectorizer.transform([cleaned])
    shap_vals = explainer.shap_values(vec)[0]
    nonzero   = vec.nonzero()[1]
    if len(nonzero) == 0:
        return
    vals  = shap_vals[nonzero]
    names = feature_names[nonzero]

    # Top 10 by absolute value
    top_idx = np.argsort(np.abs(vals))[::-1][:10]
    top_vals  = vals[top_idx]
    top_names = names[top_idx]

    # Colors: red = pushes Fake, green = pushes Real
    colors = ["#e05252" if v > 0 else "#52c48a" for v in top_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0d0d0f")
    ax.set_facecolor("#0d0d0f")

    bars = ax.barh(range(len(top_names)), top_vals, color=colors, height=0.6)

    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, color="#f5f0e8", fontsize=11)
    ax.set_xlabel("SHAP value (positive = Fake, negative = Real)", color="#c8a96e")
    ax.set_title(f"SHAP Explanation — {label}\n{text[:60]}...", color="#f5f0e8", fontsize=10)
    ax.tick_params(colors="#c8a96e")
    ax.spines["bottom"].set_color("#c8a96e")
    ax.spines["left"].set_color("#c8a96e")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(x=0, color="#c8a96e", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Saved: {filename}")

# Save plots for 2 fake + 2 real samples
save_shap_plot("Fake Example 1", samples["fake_1"], cleaned_samples["fake_1"], "shap_fake_1.png")
save_shap_plot("Fake Example 2", samples["fake_2"], cleaned_samples["fake_2"], "shap_fake_2.png")
save_shap_plot("Real Example 1", samples["real_1"], cleaned_samples["real_1"], "shap_real_1.png")
save_shap_plot("Real Example 2", samples["real_2"], cleaned_samples["real_2"], "shap_real_2.png")