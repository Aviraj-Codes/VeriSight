import re
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:3000",
    "https://verisight.vercel.app"
])

# ── Load model once at startup ────
print("🔄  Loading model pipeline...")
try:
    pipeline = joblib.load("pipeline.pkl")
    print("✅  Model loaded successfully.")
except FileNotFoundError:
    pipeline = None
    print("❌  pipeline.pkl not found. Run  python train_model.py  first.")

FAKE_THRESHOLD = 0.72

# ── Text cleaning (must match train_model.py) ───
def clean(text):
    """Lowercase, strip URLs, punctuation, and extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Routes ──
@app.route("/", methods=["GET"])
def health():
    """Simple health-check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": pipeline is not None,
        "message": "VeriSight backend is running."
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON: { "text": "..." }
    Returns JSON: { "prediction": "Fake"|"Real", "confidence": 0.87 }
    """
    # ── Validate model ───
    if pipeline is None:
        return jsonify({
            "error": "Model not loaded. Run python train_model.py first."
        }), 503
    # ── Validate request body ────
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400
    raw_text = data.get("text", "")
    if not isinstance(raw_text, str) or not raw_text.strip():
        return jsonify({"error": "Field 'text' is required and must be a non-empty string."}), 400
    if len(raw_text) > 50_000:
        return jsonify({"error": "Text is too long (max 50,000 characters)."}), 400
    # ── Predict ───
    cleaned = clean(raw_text)
    # predict_proba returns [[prob_Real, prob_Fake]]
    proba     = pipeline.predict_proba([cleaned])[0]
    prob_real = float(proba[0])
    prob_fake = float(proba[1])
    # Apply threshold — model must be confident enough to call Fake
    # otherwise default to Real to avoid false positives
    if prob_fake >= FAKE_THRESHOLD:
        prediction = "Fake"
        confidence = prob_fake
    else:
        prediction = "Real"
        confidence = prob_real
    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 4),
    })

# ── Entry point ────
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,      
    )