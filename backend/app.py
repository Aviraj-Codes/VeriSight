"""
app.py
------
Flask backend for VeriSight.
Loads the trained pipeline and exposes POST /predict.
Now includes keyword explainability — top words that influenced the prediction.

Start with:
    python app.py
"""

import re
import joblib
import numpy as np
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

# ── Confidence threshold ──────────────────────────────────────────────
FAKE_THRESHOLD = 0.50

# ── Text cleaning ─────────────────────────────────────────────────────
def clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Keyword explainability ────────────────────────────────────────────
def get_top_keywords(cleaned_text, prediction, top_n=5):
    """
    Returns the top N words from the input text that most influenced
    the prediction, using TF-IDF weights × model coefficients.
    """
    try:
        # Transform the input text to TF-IDF vector
        tfidf_matrix = vectorizer.transform([cleaned_text])
        feature_names = np.array(vectorizer.get_feature_names_out())

        # Get model coefficients for Fake class (index 1)
        # Positive coef → pushes toward Fake
        # Negative coef → pushes toward Real
        coefs = model.coef_[0]

        # Get non-zero features for this input
        nonzero_indices = tfidf_matrix.nonzero()[1]
        if len(nonzero_indices) == 0:
            return []

        # Score = tfidf_weight × coefficient
        tfidf_weights = np.array(tfidf_matrix[0, nonzero_indices]).flatten()
        coef_weights  = coefs[nonzero_indices]
        scores        = tfidf_weights * coef_weights

        # If Fake: pick highest positive scores (pushed toward Fake)
        # If Real: pick highest negative scores (pushed toward Real)
        if prediction == "Fake":
            top_idx = np.argsort(scores)[::-1][:top_n]
        else:
            top_idx = np.argsort(scores)[:top_n]

        keywords = []
        for idx in top_idx:
            word  = feature_names[nonzero_indices[idx]]
            score = float(abs(scores[idx]))
            if score > 0 and len(word) > 2:   # skip trivial words
                keywords.append({
                    "word": word,
                    "score": round(score, 4)
                })

        return keywords[:top_n]

    except Exception as e:
        print(f"Explainability error: {e}")
        return []

# ── Routes ────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": pipeline is not None,
        "message": "VeriSight backend is running.",
        "datasets": ["FakeNewsNet (PolitiFact + GossipCop)", "news_articles.csv"]
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
    cleaned   = clean(raw_text)
    proba     = pipeline.predict_proba([cleaned])[0]
    prob_real = float(proba[0])
    prob_fake = float(proba[1])

    if prob_fake >= FAKE_THRESHOLD:
        prediction = "Fake"
        confidence = prob_fake
    else:
        prediction = "Real"
        confidence = prob_real

    # ── Explainability ────────────────────────────────────────────────
    keywords = get_top_keywords(cleaned, prediction, top_n=5)

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "keywords":   keywords,
    })

# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)