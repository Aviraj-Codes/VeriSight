"""
app.py
------
Flask backend for VeriSight with SHAP explainability.
Loads the trained pipeline and exposes POST /predict.
Returns predictions + SHAP-based keyword explanations.

"""

import re
import joblib
import numpy as np
import shap
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── App setup ─────────────────────────────────────────────────────────
app = Flask(__name__)

CORS(app, origins=[
    "http://localhost:3000",
    "https://verisight-x.vercel.app"
])

# ── Load model once at startup ────────────────────────────────────────
print("🔄  Loading model pipeline...")
try:
    pipeline   = joblib.load("pipeline.pkl")
    vectorizer = pipeline.named_steps["tfidf"]
    model      = pipeline.named_steps["clf"]
    print("✅  Model loaded successfully.")
except FileNotFoundError:
    pipeline = vectorizer = model = None
    print("❌  pipeline.pkl not found. Run  python train_model.py  first.")

# ── Build SHAP explainer ──────────────────────────────────────────────
shap_explainer = None
if pipeline is not None:
    print("🔍  Building SHAP explainer (this may take 10-15 seconds)...")
    try:
        # Use a small background sample for faster SHAP computation
        background_texts = [
            "government officials announce new policy today",
            "breaking news viral story spreading online",
            "scientists discover research study findings",
            "fake hoax conspiracy theory spreading",
            "federal reserve interest rates economic news",
            "celebrity gossip entertainment news story",
            "political election voting campaign announcement",
            "healthcare medical treatment disease research",
        ]
        background_matrix = vectorizer.transform(background_texts) 
        shap_explainer = shap.LinearExplainer(
            model,
            background_matrix,
            feature_perturbation="interventional"
        )
        print("✅  SHAP explainer ready.\n")
    except Exception as e:
        print(f"⚠️   SHAP initialization failed: {e}")
        print("    Falling back to coefficient-based explanations.\n")
        shap_explainer = None

# ── Confidence threshold ──────────────────────────────────────────────
FAKE_THRESHOLD = 0.40

# ── Text cleaning ─────────────────────────────────────────────────────
def clean(text):
    """Lowercase, strip URLs, punctuation, and extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── SHAP keyword explainability ───────────────────────────────────────
def get_shap_keywords(cleaned_text, prediction, top_n=5):
    """
    Returns top N words using SHAP values.
    SHAP values show exact contribution of each word to the prediction.
    Positive SHAP → pushes toward Fake
    Negative SHAP → pushes toward Real
    """
    if shap_explainer is None:
        return []
    try:
        # Transform input text to TF-IDF vector
        vec = vectorizer.transform([cleaned_text])
        
        # Get SHAP values for this sample
        shap_vals = shap_explainer.shap_values(vec)[0]
        
        # Get non-zero feature indices
        nonzero_indices = vec.nonzero()[1]
        
        if len(nonzero_indices) == 0:
            return []

        # Get feature names
        feature_names = np.array(vectorizer.get_feature_names_out())
        
        # Extract SHAP values and names for non-zero features
        shap_vals_nz  = shap_vals[nonzero_indices]
        feature_names_nz = feature_names[nonzero_indices]

        # Select top words based on prediction direction
        # For Fake: take highest positive SHAP values
        # For Real: take highest negative SHAP values (most Real-supporting)
        if prediction == "Fake":
            # Sort by descending SHAP (most Fake-pushing)
            top_idx = np.argsort(shap_vals_nz)[::-1][:top_n]
        else:
            # Sort by ascending SHAP (most Real-pushing)
            top_idx = np.argsort(shap_vals_nz)[:top_n]
        keywords = []
        for idx in top_idx:
            word  = feature_names_nz[idx]
            score = float(abs(shap_vals_nz[idx]))
            
            # Filter out very short words
            if len(word) > 2 and score > 0.001:
                keywords.append({
                    "word":  word,
                    "score": round(score, 4)
                })
        return keywords[:top_n]

    except Exception as e:
        print(f"SHAP error: {e}")
        return []

# ── Fallback: coefficient-based keywords (if SHAP fails) ──────────────
def get_coefficient_keywords(cleaned_text, prediction, top_n=5):
    """
    Fallback method using model coefficients directly.
    Less rigorous than SHAP but faster.
    """
    try:
        tfidf_matrix = vectorizer.transform([cleaned_text])
        feature_names = np.array(vectorizer.get_feature_names_out())
        # Get model coefficients for Fake class (index 1)
        coefs = model.coef_[0]
        # Get non-zero features
        nonzero_indices = tfidf_matrix.nonzero()[1]
        if len(nonzero_indices) == 0:
            return []
        # Score = tfidf_weight × coefficient
        tfidf_weights = np.array(tfidf_matrix[0, nonzero_indices]).flatten()
        coef_weights  = coefs[nonzero_indices]
        scores        = tfidf_weights * coef_weights
        # Select based on prediction
        if prediction == "Fake":
            top_idx = np.argsort(scores)[::-1][:top_n]
        else:
            top_idx = np.argsort(scores)[:top_n]
        keywords = []
        for idx in top_idx:
            word  = feature_names[nonzero_indices[idx]]
            score = float(abs(scores[idx]))
            if len(word) > 2:
                keywords.append({
                    "word": word,
                    "score": round(score, 4)
                })
        return keywords[:top_n]
    except Exception as e:
        print(f"Coefficient error: {e}")
        return []

# ── Routes ────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    """Simple health-check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": pipeline is not None,
        "shap_enabled": shap_explainer is not None,
        "message": "VeriSight backend is running.",
        "datasets": ["FakeNewsNet (PolitiFact + GossipCop)", "news_articles.csv"],
        "explainability": "SHAP-based keyword analysis"
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON: { "text": "..." }
    Returns JSON: {
        "prediction": "Fake"|"Real",
        "confidence": 0.87,
        "keywords": [{"word": "...", "score": 0.42}, ...]
    }
    """
    # ── Validate model ────────────────────────────────────────────────
    if pipeline is None:
        return jsonify({
            "error": "Model not loaded. Run python train_model.py first."
        }), 503
    # ── Validate request ──────────────────────────────────────────────
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400
    raw_text = data.get("text", "")
    if not isinstance(raw_text, str) or not raw_text.strip():
        return jsonify({"error": "Field 'text' is required and must be non-empty."}), 400
    if len(raw_text) > 50_000:
        return jsonify({"error": "Text too long (max 50,000 characters)."}), 400
    # ── Predict ───────────────────────────────────────────────────────
    cleaned = clean(raw_text)
    # predict_proba returns [[prob_Real, prob_Fake]]
    proba     = pipeline.predict_proba([cleaned])[0]
    prob_real = float(proba[0])
    prob_fake = float(proba[1])
    # Apply confidence threshold
    if prob_fake >= FAKE_THRESHOLD:
        prediction = "Fake"
        confidence = prob_fake
    else:
        prediction = "Real"
        confidence = prob_real
    # ── Explainability ────────────────────────────────────────────────
    # Try SHAP first, fall back to coefficients if SHAP fails
    if shap_explainer is not None:
        keywords = get_shap_keywords(cleaned, prediction, top_n=5)
    else:
        keywords = get_coefficient_keywords(cleaned, prediction, top_n=5)

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "keywords":   keywords,
    })

# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)