---
title: DetectIQ
emoji: 🏃
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.19.0
python_version: '3.13'
app_file: app.py
pinned: false
license: mit
short_description: AI vs Human Text Detector
---

# DetectIQ: AI vs Human Text Detection

---

## Problem Statement

Given a passage of text, determine whether it was written by a human or generated
by an AI system. This is an increasingly important problem as AI-generated content
becomes harder to distinguish from human writing by eye alone.

---

## Dataset

- **Source:** Training data (`train_data_with_labels.xlsx`)
- **Size:** 8,176 samples — 4,088 human-written (label 0), 4,088 AI-generated (label 1)
- **Balance:** Perfect 50/50 split — no class weighting required
- **Text style:** Long-form academic/essay text averaging ~2,100 characters

---

## Machine Learning/Deep Learning Models


| # | Model | Feature Input | Tuning |
|---|-------|--------------|--------|
| 1 | SVM (LinearSVC) | TF-IDF (10K, bigrams) | GridSearchCV, C ∈ [0.1, 1, 10] |
| 2 | Decision Tree | TF-IDF (10K, bigrams) | RandomizedSearchCV (12 combos) |
| 3 | AdaBoost | TF-IDF (10K, bigrams) | GridSearchCV, n_estimators + learning_rate |
| 4 | FNN | TF-IDF dense (10K-dim) | EarlyStopping, ReduceLROnPlateau |
| 5 | LSTM | Token sequences (maxlen=300) | EarlyStopping, ReduceLROnPlateau |
| 6 | TextCNN | Token sequences (maxlen=300) | EarlyStopping, ReduceLROnPlateau |

**Feature engineering:**
- TF-IDF: 10,000 features, unigrams+bigrams, `sublinear_tf=True`, `min_df=3`
- GloVe/LSA: 100-dimensional dense embeddings averaged per document
- Linguistic: 15 stylometric signals (sentence length, vocabulary richness, Flesch score, hapax legomena ratio, punctuation density, and more)

---

## LLM Integration

Two LLMs are integrated meaningfully into the prediction pipeline:

### LLM 1 — Qwen2.5-1.5B-Instruct
**Purpose:** Generates a natural-language explanation of *why* the ML model classified
the text as AI or Human. Takes the prediction label, confidence score, and top
linguistic signals as context, then produces a 3–4 sentence explanation referencing
specific patterns in the input text.

**Why this model:** Qwen2.5-1.5B is an instruction-tuned model that runs on CPU in HF
Spaces free tier. It is capable of coherent, contextually relevant explanations. Its
instruction-following format (`<|im_start|>`) gives clean, structured output without
post-processing.

### LLM 2 — TinyLlama-1.1B-Chat-v1.0
**Purpose:** Performs a structured stylistic analysis of the text across four dimensions: sentence length variation, vocabulary diversity, hedging/academic phrasing, and overall authorship signal. Provides a complementary view to the ML feature-importance charts.

**Why this model:** TinyLlama-1.1B-Chat uses a ChatML-style prompt format and is the
most capable model that reliably fits in HF Spaces CPU memory alongside the ML models.
At 1.1B parameters it produces more coherent structured analysis than 0.5B alternatives.

---

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|-------|----------|-----------|--------|-----|---------|
| SVM | 0.980 | 0.979 | 0.980 | 0.980 | 0.997 |
| FNN | 0.980 | 0.972 | 0.988 | 0.980 | 0.997 |
| AdaBoost | 0.962 | 0.957 | 0.968 | 0.962 | 0.992 |
| Decision Tree | 0.907 | 0.910 | 0.903 | 0.907 | 0.912 |
| CNN | 0.897 | 0.863 | 0.945 | 0.902 | 0.966 |
| LSTM | 0.818 | 0.820 | 0.814 | 0.817 | 0.885 |


## What I Learned

- Integrating LLMs as explainability tools (not just classifiers) adds meaningful value
  to a prediction system. The LLM explanation often surfaces linguistic patterns that
  a feature-importance bar chart doesn't convey to a non-technical user
- Model selection for inference on constrained hardware (HF Spaces CPU) requires careful attention to parameter count, quantization support, and memory footprint
- TF-IDF remains a strong baseline for authorship attribution — word-choice distributions differ dramatically between human and AI writing at corpus scale
- POS-aware lemmatization meaningfully reduces vocabulary size vs. noun-only lemmatization

---

## Project Structure

```
detectiq/
├── app.py                  
├── requirements.txt        
├── README.md              
└── models/                 
    ├── svm_model.pkl
    ├── decision_tree_model.pkl
    ├── adaboost_model.pkl
    ├── fnn_model.h5
    ├── lstm_model.h5
    ├── cnn_model.h5
    ├── tfidf_vectorizer.pkl
    ├── tokenizer.pkl
    ├── ling_scaler.pkl
    └── model_results.json
```

---

## How to Run Locally

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web application framework |
| `transformers` | HuggingFace LLM loading and inference |
| `torch` | PyTorch backend for LLMs |
| `scikit-learn` | ML models, TF-IDF, metrics |
| `tensorflow` | FNN, LSTM, CNN inference |
| `nltk` | Tokenization, POS tagging, lemmatization |
| `accelerate` | Efficient model loading on CPU/GPU |
| `pdfplumber` / `PyPDF2` | PDF text extraction |
| `python-docx` | Word document extraction |