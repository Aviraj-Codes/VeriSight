# VeriSight — AI-Powered Fake News Detector

> Analyze news authenticity in seconds using machine learning.

VeriSight is a full-stack web application that uses a trained machine learning model to classify news articles as **Real** or **Fake**, along with a confidence score. It features a newspaper-themed glassmorphism UI built in React, backed by a Flask REST API.

---

## Features

- Paste any news article or headline and get an instant prediction
- Confidence score displayed as an animated progress bar
- Real → green result, Fake → red result with clear visual feedback
- Dark editorial UI with glassmorphism card and newspaper background
- Trained on a real-world dataset of 2,095 labeled news articles
- TF-IDF + Logistic Regression pipeline with ~80% accuracy

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React (Create React App), plain CSS |
| Backend | Python, Flask, Flask-CORS |
| ML Model | scikit-learn — TF-IDF + Logistic Regression |
| Data | Custom news articles dataset (CSV) |

---

## Project Structure

```
VeriSight/
├── backend/
│   ├── app.py                  # Flask API — POST /predict endpoint
│   ├── train_model.py          # Training script — run once to generate model
│   ├── news_articles.csv       # Labeled dataset (Fake / Real)
│   ├── requirements.txt        # Python dependencies
│   ├── model.pkl               # Saved model (generated after training)
│   ├── vectorizer.pkl          # Saved vectorizer (generated after training)
│   └── pipeline.pkl            # Full pipeline (generated after training)
│
├── frontend/
│   ├── public/
│   │   ├── index.html
│   │   └── favicon.png         # VeriSight logo (browser tab icon)
│   ├── src/
│   │   ├── App.js              # Main React component
│   │   ├── App.css             # All styles
│   │   └── index.js            # React entry point
│   └── package.json
│
├── .gitignore
└── README.md
```

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | Logistic Regression |
| Features | TF-IDF (unigrams + bigrams) |
| Vocabulary size | 30,000 tokens |
| Training samples | ~1,281 |
| Test accuracy | ~80% |
| Fake threshold | 0.65 (tuned to reduce false positives) |

The dataset was balanced before training (801 Fake, 801 Real) to prevent the model from being biased toward predicting Fake. A confidence threshold of 0.65 is applied at prediction time — the model must be at least 65% confident to label something as Fake.

---

## Known Limitations

- The model is trained on a relatively small dataset (~2,000 articles) — accuracy may vary on very niche or recent topics
- The model works best on English-language articles
- Short inputs (single words or very short phrases) may produce unreliable results
- The model detects patterns from training data and is not a substitute for professional fact-checking

---

## Future Improvements

- [ ] Add support for URL input (auto-fetch article text)
- [ ] Upgrade to a larger pre-trained model (e.g. BERT)
- [ ] Add article history / past results
- [ ] Deploy backend to Render or Railway
- [ ] Deploy frontend to Vercel or Netlify

---

## License

This project is for educational purposes.

---

## Author

Built by **Aviraj Bandyopadhyay** · [GitHub](https://github.com/Aviraj-Codes)