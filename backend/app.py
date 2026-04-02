import re
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:3000",
    "https://verisight-x.vercel.app"
])

# ── Load model ──
print("🔄 Loading model pipeline...")
try:
    pipeline = joblib.load("pipeline.pkl")
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    pipeline = None
    print("❌ pipeline.pkl not found. Run python train_model.py first.")

# ── Tuned thresholds ──
FAKE_THRESHOLD = 0.60
UNCERTAIN_LOW = 0.45
UNCERTAIN_HIGH = 0.60

# ── Cleaning (same as training) ──
def clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Health route ──
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": pipeline is not None,
        "message": "VeriSight backend is running."
    })

# ── Prediction route ──
@app.route("/predict", methods=["POST"])
def predict():

    if pipeline is None:
        return jsonify({
            "error": "Model not loaded. Run python train_model.py first."
        }), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    raw_text = data.get("text", "")
    if not isinstance(raw_text, str) or not raw_text.strip():
        return jsonify({"error": "Field 'text' must be a non-empty string."}), 400

    if len(raw_text) > 50000:
        return jsonify({"error": "Text too long (max 50,000 chars)."}), 400

    # ── Clean input ──
    cleaned = clean(raw_text)

    # ── Model prediction ──
    proba = pipeline.predict_proba([cleaned])[0]
    prob_real = float(proba[0])
    prob_fake = float(proba[1])

    word_count = len(cleaned.split())

    # ── Decision logic ──

    # Case 1: short claims (different behavior)
    if word_count < 8:
        if prob_fake > 0.35:
            prediction = "Fake"
            confidence = prob_fake
        else:
            prediction = "Real"
            confidence = prob_real

    # Case 2: uncertainty band
    elif UNCERTAIN_LOW < prob_fake < UNCERTAIN_HIGH:
        return jsonify({
            "prediction": "Uncertain",
            "confidence": round(prob_fake, 4),
            "prob_fake": round(prob_fake, 4),
            "prob_real": round(prob_real, 4)
        })

    # Case 3: confident prediction
    elif prob_fake >= FAKE_THRESHOLD:
        prediction = "Fake"
        confidence = prob_fake
    else:
        prediction = "Real"
        confidence = prob_real

    # ── Debug logging ──
    print("-----")
    print(f"TEXT: {raw_text[:100]}")
    print(f"Fake: {prob_fake:.4f}, Real: {prob_real:.4f}")
    print(f"Prediction: {prediction}")

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "prob_fake": round(prob_fake, 4),
        "prob_real": round(prob_real, 4)
    })

# ── Run server ──
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)