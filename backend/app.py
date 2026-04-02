import re
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Load model ──
print("🔄 Loading model...")
try:
    pipeline = joblib.load("pipeline.pkl")
    print("✅ Model loaded")
except:
    pipeline = None
    print("❌ Model not found")

# ── CLEAN (same as training) ──
def clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Health ──
@app.route("/")
def home():
    return jsonify({"status": "ok", "model_loaded": pipeline is not None})

# ── Predict ──
@app.route("/predict", methods=["POST"])
def predict():

    if pipeline is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "Empty input"}), 400

    cleaned = clean(text)

    proba = pipeline.predict_proba([cleaned])[0]
    prob_real = float(proba[0])
    prob_fake = float(proba[1])

    # ── SIMPLE DECISION ──
    if prob_fake > 0.5:
        prediction = "Fake"
        confidence = prob_fake
    else:
        prediction = "Real"
        confidence = prob_real

    # Debug log
    print("-----")
    print(f"Input: {text[:100]}")
    print(f"Fake: {prob_fake:.4f}, Real: {prob_real:.4f}")
    print(f"Prediction: {prediction}")

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "prob_fake": round(prob_fake, 4),
        "prob_real": round(prob_real, 4)
    })

# ── Run ──
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)