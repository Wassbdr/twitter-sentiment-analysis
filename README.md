# Twitter Sentiment Analysis — Impact on Stock Prices

Benchmark study: does Twitter/X sentiment predict stock price movements? Compares 7 NLP models (DistilBERT, GPT-2, and 5 others) on sentiment classification + correlation analysis with market data.

---

## What this project does

1. **Collects** Twitter data related to specific stocks (AAPL, TSLA, …)
2. **Classifies** tweet sentiment using multiple NLP models
3. **Correlates** sentiment scores with next-day stock price changes
4. **Visualises** results via an interactive Streamlit dashboard

## Models Benchmarked

| Model | Type | F1 Score |
|---|---|---|
| DistilBERT (fine-tuned) | Transformer | **0.79** |
| GPT-2 (zero-shot) | Generative LM | — |
| VADER | Lexicon-based | — |
| TextBlob | Rule-based | — |
| Naive Bayes | Classical ML | — |
| Logistic Regression (TF-IDF) | Classical ML | — |
| SVM (TF-IDF) | Classical ML | — |

Best single model: **DistilBERT** at F1 = 0.79

## Quick Start

```bash
pip install -r requirements.txt

# 1. Run the data pipeline + GPT-2 generation (in notebook)
jupyter notebook src/data.ipynb

# 2. Launch Streamlit dashboard
streamlit run src/app.py
```

## Project Structure

```
twitter-sentiment-analysis/
├── src/
│   ├── data.ipynb      # Data loading, preprocessing, model benchmarks
│   └── app.py          # Streamlit dashboard
└── requirements.txt
```

## Tech Stack

- `transformers` (DistilBERT, GPT-2)
- `scikit-learn` (classical models)
- `vaderSentiment`, `textblob`
- `pandas`, `matplotlib`, `plotly`
- `streamlit`

## License

MIT
