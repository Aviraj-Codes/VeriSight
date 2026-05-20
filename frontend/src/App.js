import { useState } from "react";
import "./App.css";

function App() {
  const [inputText, setInputText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!inputText.trim()) return;

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const response = await fetch("https://verisight-backend.onrender.com/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: inputText }),
      });

      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to connect to the server. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const confidencePct = result ? Math.round(result.confidence * 100) : 0;
  const isFake = result?.prediction?.toLowerCase() === "fake";

  // Normalize keyword scores for bar width (0–100%)
  const maxScore = result?.keywords?.length
    ? Math.max(...result.keywords.map((k) => k.score))
    : 1;

  return (
    <div className="app-wrapper">
      <div className="overlay" />
      <div className="ink ink-1" />
      <div className="ink ink-2" />

      <div className="card">
        {/* Header */}
        <div className="card-header">
          <div className="badge">PRESS INTELLIGENCE</div>
          <img src="/favicon.png" alt="VeriSight Logo" className="card-logo" />
          <h1 className="title">VeriSight</h1>
          <p className="subtitle">Analyze news authenticity with AI</p>
          <div className="divider" />
        </div>

        {/* Input */}
        <div className="input-section">
          <label className="input-label">PASTE ARTICLE OR HEADLINE</label>
          <textarea
            className="textarea"
            placeholder="Enter the news article text here to verify its authenticity..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            rows={6}
          />
        </div>

        {/* Button */}
        <button
          className={`analyze-btn ${loading ? "loading" : ""}`}
          onClick={handleAnalyze}
          disabled={loading || !inputText.trim()}
        >
          {loading ? (
            <span className="btn-content">
              <span className="spinner" />
              Analyzing...
            </span>
          ) : (
            <span className="btn-content">
              <span className="btn-icon">⬡</span>
              Run Analysis
            </span>
          )}
        </button>

        {/* Error */}
        {error && (
          <div className="error-box">
            <span className="error-icon">⚠</span>
            <span>{error}</span>
          </div>
        )}

        {/* Result */}
        {result && (
          <div className={`result-box ${isFake ? "result-fake" : "result-real"}`}>

            {/* Verdict header */}
            <div className="result-header">
              <span className="result-label">VERDICT</span>
              <span className={`result-badge ${isFake ? "badge-fake" : "badge-real"}`}>
                {isFake ? "⚠ FAKE" : "✓ REAL"}
              </span>
            </div>

            {/* Prediction word */}
            <div className="result-prediction">
              <span className="prediction-word" data-fake={isFake}>
                {result.prediction}
              </span>
              <span className="prediction-sub">content detected</span>
            </div>

            {/* Confidence bar */}
            <div className="confidence-section">
              <div className="confidence-row">
                <span className="confidence-label">CONFIDENCE SCORE</span>
                <span className="confidence-pct">{confidencePct}%</span>
              </div>
              <div className="progress-track">
                <div
                  className={`progress-fill ${isFake ? "fill-fake" : "fill-real"}`}
                  style={{ width: `${confidencePct}%` }}
                />
                <div className="progress-glow" style={{ left: `${confidencePct}%` }} />
              </div>
              <div className="confidence-scale">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
              </div>
            </div>

            {/* Keyword explainability */}
            {result.keywords && result.keywords.length > 0 && (
              <div className="keywords-section">
                <div className="keywords-header">
                  <span className="keywords-label">
                    {isFake ? "⚡ KEY SIGNALS DRIVING THIS VERDICT" : "✦ KEY SIGNALS SUPPORTING AUTHENTICITY"}
                  </span>
                </div>
                <div className="keywords-list">
                  {result.keywords.map((kw, i) => (
                    <div className="keyword-row" key={i}>
                      <span className="keyword-word">{kw.word}</span>
                      <div className="keyword-bar-track">
                        <div
                          className={`keyword-bar-fill ${isFake ? "kbar-fake" : "kbar-real"}`}
                          style={{ width: `${(kw.score / maxScore) * 100}%` }}
                        />
                      </div>
                      <span className="keyword-score">{(kw.score * 100).toFixed(1)}</span>
                    </div>
                  ))}
                </div>
                <p className="keywords-note">
                  These words had the highest influence on the model's decision.
                </p>
              </div>
            )}

            {/* Result note */}
            <p className="result-note">
              {isFake
                ? "This content shows patterns commonly associated with misinformation. Exercise caution before sharing."
                : "This content appears to align with authentic reporting patterns. Always verify with primary sources."}
            </p>
          </div>
        )}

        {/* Footer */}
        <div className="card-footer">
          <span>Trained on FakeNewsNet · PolitiFact · GossipCop</span>
        </div>
      </div>
    </div>
  );
}

export default App;