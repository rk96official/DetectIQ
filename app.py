import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import re, string, pickle, json, io, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="DetectIQ",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display:none !important; }
#MainMenu, footer, header[data-testid="stHeader"] { display:none !important; }
.block-container  { padding-top: 1rem; max-width: 1180px; }

/* ---- Top navigation bar ---- */
.st-key-topnav {
    border-bottom:1px solid #e6e9f0; margin-bottom:6px; padding:2px 0 8px;
}
.st-key-topnav [data-testid="stHorizontalBlock"] { align-items:center; }
.st-key-topnav .stButton > button {
    border:none; background:transparent; color:#475065; font-weight:600;
    font-size:.95rem; padding:6px 4px; box-shadow:none;
}
.st-key-topnav .stButton > button:hover { color:#0D9488; background:transparent; }
/* brand — target the label <p> inside the button, not just the button */
.st-key-nav_brand button,
.st-key-nav_brand button p,
.st-key-nav_brand [data-testid="stMarkdownContainer"] p {
    color:#1B2A4A !important; font-weight:800 !important;
    font-size:1.9rem !important; letter-spacing:-.3px !important;
    line-height:1.1 !important;
}
/* active nav item (primary) */
.st-key-topnav .stButton > button[kind="primary"] {
    color:#0D9488; background:transparent; border-bottom:2px solid #0D9488;
    border-radius:0;
}

/* ---- Home hero (left column) ---- */
.hero-text       { padding:34px 8px 0; }
.hero-eyebrow    { color:#0D9488; font-weight:700; font-size:.82rem;
                   text-transform:uppercase; letter-spacing:1.5px; }
.hero-text h1    { color:#1B2A4A; font-size:2.7rem; line-height:1.12; margin:10px 0 0;
                   font-weight:800; letter-spacing:-.5px; }
.hero-text h1 span { color:#0D9488; }
.hero-text p     { color:#5a6478; font-size:1.05rem; line-height:1.55; margin:18px 0 0; }
.try-it          { display:flex; align-items:center; gap:10px; margin-top:30px;
                   color:#1B2A4A; font-weight:700; font-size:1.05rem; }
.try-it .arrow   { font-size:1.6rem; color:#0D9488; animation:nudge 1.3s ease-in-out infinite; }
@keyframes nudge { 0%,100%{transform:translateX(0)} 50%{transform:translateX(8px)} }
.feat-row        { display:flex; gap:22px; margin-top:34px; flex-wrap:wrap; }
.feat            { color:#5a6478; font-size:.9rem; }
.feat b          { color:#1B2A4A; display:block; font-size:1.35rem; }

/* ---- Input card ---- */
.st-key-trybox, .st-key-pagebox {
    border:1px solid #e3e8f0; border-radius:18px; padding:8px 20px 18px;
    box-shadow:0 10px 40px rgba(27,42,74,.08); background:#fff;
}
.st-key-trybox .stTextArea textarea, .st-key-pagebox .stTextArea textarea {
    border-radius:12px; font-size:.95rem;
}

/* ---- Original hero (kept for inner page headers) ---- */
.hero            { background:linear-gradient(135deg,#1B2A4A,#2d4a7c); color:#fff;
                   border-radius:16px; padding:22px 28px; margin-bottom:18px; }
.hero h1         { color:#fff; font-size:2.0rem; margin:0; letter-spacing:.3px; }
.hero p          { color:#cdd6e8; margin:6px 0 0; font-size:.95rem; }
/* ---- Verdict card ---- */
.verdict-card    { border-radius:18px; padding:24px 28px; color:#fff; margin:6px 0 4px;
                   box-shadow:0 8px 26px rgba(0,0,0,0.15); }
.v-ai            { background:linear-gradient(135deg,#c92a2a,#ff6b6b); }
.v-human         { background:linear-gradient(135deg,#2b8a3e,#51cf66); }
.verdict-top     { display:flex; justify-content:space-between; align-items:center;
                   flex-wrap:wrap; gap:8px; }
.verdict-label   { font-size:2.2rem; font-weight:800; letter-spacing:.5px; }
.model-pill      { background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.45);
                   border-radius:999px; padding:6px 16px; font-size:.82rem; font-weight:600;
                   white-space:nowrap; }
.conf-wrap       { margin-top:20px; }
.conf-row        { display:flex; justify-content:space-between; align-items:baseline;
                   font-size:.9rem; font-weight:600; opacity:.97; }
.conf-pct        { font-size:1.15rem; font-weight:800; }
.conf-track      { height:13px; border-radius:8px; background:rgba(255,255,255,.28);
                   margin-top:8px; overflow:hidden; }
.conf-fill       { height:100%; background:rgba(255,255,255,.95); border-radius:8px; }
.verdict-foot    { font-size:.82rem; opacity:.9; margin-top:12px; }

/* ---- Stat boxes ---- */
.stat-box        { background:#f0f4ff; border-radius:12px; padding:14px 16px;
                   text-align:center; border:1px solid #dde3f0; }
.stat-number     { font-size:1.6rem; font-weight:700; color:#1B2A4A; }
.stat-label      { font-size:0.78rem; color:#888; text-transform:uppercase;
                   letter-spacing:.5px; }

/* ---- LLM panels ---- */
.llm-box         { background:#f8fafc; border:1px solid #e6edf3;
                   border-left:4px solid #0D9488; border-radius:10px;
                   padding:16px 18px; height:100%; }
.llm-head        { font-weight:700; color:#0D9488; margin-bottom:2px; }
.llm-sub         { font-size:.8rem; color:#8a93a3; margin-bottom:10px; }

/* ---- Per-model mini verdict cards (comparison) ---- */
.mini-card       { background:#fff; border:1px solid #e6edf3; border-radius:14px;
                   padding:16px 12px; text-align:center; height:100%;
                   box-shadow:0 2px 10px rgba(0,0,0,.05); }
.mini-model      { font-weight:700; color:#1B2A4A; font-size:.95rem; }
.mini-type       { font-size:.68rem; color:#9aa3b2; text-transform:uppercase;
                   letter-spacing:.4px; margin-bottom:8px; }
.mini-verdict    { font-weight:800; font-size:1.0rem; margin:4px 0 4px; }
.mini-conf       { font-size:.78rem; color:#888; }
.consensus-card  { border-radius:16px; padding:18px 24px; color:#fff; margin:4px 0 14px;
                   box-shadow:0 6px 22px rgba(0,0,0,0.12); }
.consensus-head  { font-size:.82rem; text-transform:uppercase; letter-spacing:1px;
                   opacity:.9; }
.consensus-main  { font-size:1.7rem; font-weight:800; margin-top:2px; }
</style>
""", unsafe_allow_html=True)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

import nltk
for pkg in ("punkt", "stopwords", "wordnet", "punkt_tab", "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng"):
    nltk.download(pkg, quiet=True)
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


#  Text utilities
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"): return wordnet.ADJ
    if treebank_tag.startswith("V"): return wordnet.VERB
    if treebank_tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\b\d+\b", "", text)
    nopunc = "".join(c for c in text if c not in string.punctuation)
    stop_custom = STOP_WORDS | {"et", "al", "fig", "eq", "u", "ur", "im"}
    tokens = nopunc.split()
    tokens = [t for t in tokens if t.lower() not in stop_custom]
    pos_tags = nltk.pos_tag(tokens)
    lemmatized = [LEMMATIZER.lemmatize(t, get_wordnet_pos(p)) for t, p in pos_tags]
    return " ".join(lemmatized)


def linguistic_features(text: str) -> np.ndarray:
    from collections import Counter
    sentences   = sent_tokenize(text)
    words       = word_tokenize(text.lower())
    words_alpha = [w for w in words if w.isalpha()]
    n_words = max(len(words_alpha), 1)
    n_sents = max(len(sentences), 1)
    n_chars = max(len(text), 1)

    avg_sent_len   = n_words / n_sents
    avg_word_len   = np.mean([len(w) for w in words_alpha]) if words_alpha else 0
    vocab_richness = len(set(words_alpha)) / n_words
    punct_density  = sum(1 for c in text if c in string.punctuation) / n_chars
    comma_rate     = text.count(",") / n_sents
    semicolon_rate = text.count(";") / n_sents
    question_rate  = text.count("?") / n_sents
    exclaim_rate   = text.count("!") / n_sents
    para_count     = text.count("\n") + 1
    long_word_ratio= sum(1 for w in words_alpha if len(w) > 7) / n_words
    uppercase_ratio= sum(1 for c in text if c.isupper()) / n_chars
    syllables = sum(max(1, len(re.findall(r"[aeiou]", w, re.I))) for w in words_alpha)
    flesch         = 206.835 - 1.015*(n_words/n_sents) - 84.6*(syllables/n_words)
    freq = Counter(words_alpha)
    hapax_ratio    = sum(1 for v in freq.values() if v == 1) / n_words
    avg_sent_chars = n_chars / n_sents

    return np.array([[
        avg_sent_len, avg_word_len, vocab_richness, punct_density,
        comma_rate, semicolon_rate, question_rate, exclaim_rate,
        para_count, long_word_ratio, uppercase_ratio,
        flesch, hapax_ratio, avg_sent_chars, n_words
    ]])

LING_FEATURE_NAMES = [
    "Avg sentence length (words)", "Avg word length", "Vocabulary richness (TTR)",
    "Punctuation density", "Comma rate / sentence", "Semicolon rate / sentence",
    "Question mark rate", "Exclamation rate", "Paragraph count",
    "Long-word ratio (>7 chars)", "Uppercase ratio", "Flesch Reading Ease",
    "Hapax legomena ratio", "Avg sentence length (chars)", "Total word count",
]


def compute_text_stats(text: str) -> dict:
    sentences    = sent_tokenize(text)
    words        = word_tokenize(text)
    words_alpha  = [w for w in words if w.isalpha()]
    sent_lengths = [len(word_tokenize(s)) for s in sentences] if sentences else [0]
    unique       = set(w.lower() for w in words_alpha)
    return {
        "word_count":            len(words_alpha),
        "sentence_count":        len(sentences),
        "unique_word_count":     len(unique),
        "vocab_richness":        len(unique) / max(len(words_alpha), 1),
        "avg_sentence_length":   float(np.mean(sent_lengths)),
        "median_sentence_length":float(np.median(sent_lengths)),
        "sentence_lengths":      sent_lengths,
        "avg_word_length":       float(np.mean([len(w) for w in words_alpha])) if words_alpha else 0,
    }


def extract_text_from_pdf(file) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(file) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception:
        pass
    try:
        from PyPDF2 import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(file).pages)
    except Exception as e:
        return f"[PDF extraction failed: {e}]"


def extract_text_from_docx(file) -> str:
    try:
        from docx import Document
        return "\n".join(p.text for p in Document(file).paragraphs)
    except Exception as e:
        return f"[DOCX extraction failed: {e}]"


@st.cache_resource(show_spinner=False)
def load_single_llm(model_id: str, max_new_tokens: int):
    from transformers import pipeline
    import torch

    if torch.cuda.is_available():
        return pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    return pipeline(
        "text-generation",
        model=model_id,
        device=-1,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )


def get_llm1():
    return load_single_llm("Qwen/Qwen2.5-1.5B-Instruct", 300)


def get_llm2():
    return load_single_llm("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 280)

def stream_llm(llm, prompt: str, max_new_tokens: int = 300):
    if llm is None:
        yield "LLM not available."
        return

    from threading import Thread
    from transformers import TextIteratorStreamer

    tok, model = llm.tokenizer, llm.model
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens,
                      do_sample=False, pad_token_id=tok.eos_token_id)

    err = {}
    def run_generation():
        try:
            model.generate(**gen_kwargs)
        except Exception as e:
            err["e"] = e
            streamer.text_queue.put(streamer.stop_signal)

    thread = Thread(target=run_generation, daemon=True)
    thread.start()
    try:
        for chunk in streamer:
            yield chunk
    finally:
        thread.join()
    if err:
        yield f"\n\n[LLM error: {err['e']}]"


def stream_with_spinner(gen, label: str) -> str:
    with st.spinner(label):
        first = next(gen, "")

    def stream_rest():
        if first:
            yield first
        yield from gen

    return st.write_stream(stream_rest())


def build_explain_prompt(text: str, label_str: str, confidence: float,
                         top_features: list) -> str:
    snippet = text[:400].strip().replace("\n", " ")
    features_str = "\n".join(f"  - {n}: {v:.3f}" for n, v in top_features[:5])
    return f"""<|im_start|>system
You are an expert in computational linguistics and AI text detection. Give concise, informative explanations.
<|im_end|>
<|im_start|>user
A machine learning classifier predicted this text is {label_str} with {confidence*100:.0f}% confidence.

Text excerpt: "{snippet}"

Key linguistic signals detected:
{features_str}

In 3-4 sentences, explain what linguistic patterns most likely caused this prediction. Be specific about the text, not generic.
<|im_end|>
<|im_start|>assistant
"""


def build_style_prompt(text: str, stats: dict) -> str:
    snippet = text[:500].strip().replace("\n", " ")
    return (
        "<|system|>\n"
        "You are a writing style analyst. Analyze text for AI vs. human authorship signals.</s>\n"
        "<|user|>\n"
        f"Analyze this text excerpt for stylistic patterns that indicate AI or human authorship.\n\n"
        f"Text: \"{snippet}\"\n\n"
        f"Stats: {stats['word_count']} words, {stats['sentence_count']} sentences, "
        f"vocab richness {stats['vocab_richness']:.3f}, avg sentence length {stats['avg_sentence_length']:.1f} words.\n\n"
        "Provide a structured analysis covering:\n"
        "1. Sentence length variation (uniform = AI, varied = human)\n"
        "2. Vocabulary diversity and word choice\n"
        "3. Hedging language and academic phrasing\n"
        "4. Overall authorship signal\n\n"
        "Keep each point to one sentence.</s>\n"
        "<|assistant|>\n"
    )


def build_consensus_prompt(text: str, rows: list,
                           ai_votes: int, human_votes: int) -> str:
    snippet = text[:400].strip().replace("\n", " ")
    votes_str = "\n".join(
        f"  - {r['Model']} ({r['Type']}): {r['Prediction']} at {r['Confidence']}"
        for r in rows
    )
    if ai_votes > human_votes:
        consensus = "AI-Generated"
    elif human_votes > ai_votes:
        consensus = "Human-Written"
    else:
        consensus = "a tie (split decision)"
    return f"""<|im_start|>system
You are an expert in AI text detection. Summarize how an ensemble of models voted, in clear plain language for a non-technical reader.
<|im_end|>
<|im_start|>user
Six machine-learning models each independently classified the SAME text as AI-Generated or Human-Written.

Vote tally: {ai_votes} AI, {human_votes} Human (consensus: {consensus}).

Per-model votes:
{votes_str}

Text excerpt: "{snippet}"

In 3-4 sentences, give the overall verdict, say how strong the agreement is, and note whether traditional ML and deep-learning models disagreed and what that might suggest. Do not just re-list every model.
<|im_end|>
<|im_start|>assistant
"""


@st.cache_resource(show_spinner="Loading ML/DL models...")
def load_ml_models():
    loaded = {}
    errors = []

    for key, fname in [
        ("tfidf",          "tfidf_vectorizer.pkl"),
        ("scaler",         "ling_scaler.pkl"),
        ("tokenizer",      "tokenizer.pkl"),
        ("SVM",            "svm_model.pkl"),
        ("Decision Tree",  "decision_tree_model.pkl"),
        ("AdaBoost",       "adaboost_model.pkl"),
    ]:
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            loaded[key] = pickle.load(open(path, "rb"))
        else:
            errors.append(f"{fname} not found")

    try:
        import tensorflow as tf
        for key, fname in [("FNN","fnn_model.h5"),("LSTM","lstm_model.h5"),("CNN","cnn_model.h5")]:
            path = os.path.join(MODELS_DIR, fname)
            if os.path.exists(path):
                loaded[key] = tf.keras.models.load_model(path, compile=False)
            else:
                errors.append(f"{fname} not found")
    except Exception as e:
        errors.append(f"TF models: {e}")

    res_path = os.path.join(MODELS_DIR, "model_results.json")
    if os.path.exists(res_path):
        loaded["results"] = json.load(open(res_path))

    return loaded, errors


MAX_LEN = 300

def predict(text: str, model_name: str, models: dict) -> dict:
    cleaned = clean_text(text)

    if model_name in ("SVM", "Decision Tree", "AdaBoost", "FNN"):
        if "tfidf" not in models:
            return {"error": "TF-IDF vectorizer not loaded."}
        X = models["tfidf"].transform([cleaned])
        if model_name == "FNN":
            X = X.toarray().astype("float32")
    else:
        if "tokenizer" not in models:
            return {"error": "Tokenizer not loaded."}
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seqs = models["tokenizer"].texts_to_sequences([text])
        X = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

    mdl = models.get(model_name)
    if mdl is None:
        return {"error": f"Model '{model_name}' not loaded."}

    if model_name in ("SVM", "Decision Tree", "AdaBoost"):
        label = int(mdl.predict(X)[0])
        prob  = float(mdl.predict_proba(X)[0][1])
    else:
        prob  = float(mdl(X, training=False).numpy().flatten()[0])
        label = int(prob > 0.5)

    confidence = prob if label == 1 else 1 - prob

    ling_raw = linguistic_features(text)
    ling_scaled = models["scaler"].transform(ling_raw)[0] if "scaler" in models else ling_raw[0]

    return {
        "label":     label,
        "label_str": "AI-Generated" if label == 1 else "Human-Written",
        "prob_ai":   prob,
        "confidence":confidence,
        "ling_raw":  ling_raw[0],
        "ling_scaled":ling_scaled,
        "cleaned":   cleaned,
    }


def get_top_features(pred: dict, n: int = 5) -> list:
    order = np.argsort(-np.abs(pred["ling_scaled"]))[:n]
    return [(LING_FEATURE_NAMES[i], float(pred["ling_raw"][i])) for i in order]


def render_ml_explanation(model_name: str, text: str, pred: dict, models: dict):
    st.markdown("##### ML Feature Explanation")
    if model_name == "SVM":
        mdl = models.get("SVM")
        tfidf_vec = models.get("tfidf")
        coef = None
        if mdl and hasattr(mdl, "coef_"):
            coef = mdl.coef_.flatten()
        elif mdl and hasattr(mdl, "calibrated_classifiers_"):
            try:
                coef = mdl.calibrated_classifiers_[0].estimator.coef_.flatten()
            except Exception:
                pass
        if coef is not None and tfidf_vec is not None:
            fnames  = np.array(tfidf_vec.get_feature_names_out())
            doc_vec = tfidf_vec.transform([clean_text(text)]).toarray()[0]
            active  = np.where(doc_vec > 0)[0]
            if len(active) > 0:
                contrib = coef[active] * doc_vec[active]
                order   = np.argsort(contrib)
                top_pos = active[order[-8:][::-1]]
                top_neg = active[order[:8]]
                fig, axes = plt.subplots(1, 2, figsize=(8, 3))
                axes[0].barh(fnames[top_pos][::-1], contrib[order[-8:][::-1]][::-1], color="tomato")
                axes[0].set_title("Top AI-pushing words")
                axes[1].barh(fnames[top_neg], contrib[order[:8]], color="steelblue")
                axes[1].set_title("Top Human-pushing words")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                return
        st.info("Coefficient explanation unavailable.")

    elif model_name in ("Decision Tree", "AdaBoost"):
        mdl = models.get(model_name)
        tfidf_vec = models.get("tfidf")
        if mdl and tfidf_vec and hasattr(mdl, "feature_importances_"):
            imp = mdl.feature_importances_
            fnames = np.array(tfidf_vec.get_feature_names_out())
            top = np.argsort(imp)[-10:][::-1]
            fig, ax = plt.subplots(figsize=(6, 3))
            color = "mediumseagreen" if model_name == "Decision Tree" else "darkorange"
            ax.bar(range(10), imp[top], color=color)
            ax.set_xticks(range(10))
            ax.set_xticklabels(fnames[top], rotation=45, ha="right", fontsize=8)
            ax.set_title(f"{model_name} — Global Feature Importances")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            return
        st.info("Feature importance unavailable.")

    else:
        ling_scaled = pred["ling_scaled"]
        order = np.argsort(-np.abs(ling_scaled))[:6]
        fig, ax = plt.subplots(figsize=(7, 3))
        names  = [LING_FEATURE_NAMES[i] for i in order]
        vals   = [ling_scaled[i] for i in order]
        colors = ["tomato" if v > 0 else "steelblue" for v in vals]
        ax.barh(names[::-1], vals[::-1], color=colors[::-1])
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("Std deviations from dataset mean")
        ax.set_title(f"{model_name} — Most Unusual Stylometric Signals")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("DL models don't expose per-word weights — showing linguistic deviation instead.")


def render_text_stats(text: str):
    stats = compute_text_stats(text)
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip(
        [c1, c2, c3, c4],
        ["Word Count", "Sentences", "Unique Words", "Vocab Richness"],
        [stats["word_count"], stats["sentence_count"],
         stats["unique_word_count"], f"{stats['vocab_richness']:.3f}"]
    ):
        col.markdown(
            f'<div class="stat-box"><div class="stat-number">{val}</div>'
            f'<div class="stat-label">{label}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("")
    fig, ax = plt.subplots(figsize=(7, 2.6))
    lengths = stats["sentence_lengths"]
    ax.hist(lengths, bins=max(5, min(20, len(set(lengths)))), color="#1B2A4A", alpha=0.75, edgecolor="white")
    ax.axvline(stats["avg_sentence_length"], color="tomato", linestyle="--", lw=1.5,
               label=f"Mean = {stats['avg_sentence_length']:.1f}")
    ax.axvline(stats["median_sentence_length"], color="darkgreen", linestyle=":", lw=1.5,
               label=f"Median = {stats['median_sentence_length']:.1f}")
    ax.set_xlabel("Sentence length (words)")
    ax.set_ylabel("Count")
    ax.set_title("Sentence Length Distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    return stats


def generate_report(text, pred, model_name, stats, llm_explanation, llm_style) -> bytes:
    from datetime import datetime
    buf = io.StringIO()
    buf.write("DetectIQ — AI vs Human Text Analysis Report\n")
    buf.write("=" * 55 + "\n")
    buf.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
    buf.write(f"Model         : {model_name}\n")
    buf.write(f"Prediction    : {pred['label_str']}\n")
    buf.write(f"Confidence    : {pred['confidence']*100:.1f}%\n\n")
    buf.write("Text Statistics\n" + "-"*30 + "\n")
    buf.write(f"  Word count              : {stats['word_count']}\n")
    buf.write(f"  Sentence count          : {stats['sentence_count']}\n")
    buf.write(f"  Unique words            : {stats['unique_word_count']}\n")
    buf.write(f"  Vocabulary richness     : {stats['vocab_richness']:.4f}\n")
    buf.write(f"  Avg sentence length     : {stats['avg_sentence_length']:.2f} words\n\n")
    buf.write("LLM 1 — Prediction Explanation\n" + "-"*30 + "\n")
    buf.write(llm_explanation + "\n\n")
    buf.write("LLM 2 — Style Analysis\n" + "-"*30 + "\n")
    buf.write(llm_style + "\n\n")
    buf.write("Linguistic Features\n" + "-"*30 + "\n")
    for name, val in zip(LING_FEATURE_NAMES, pred["ling_raw"]):
        buf.write(f"  {name:<40s}: {val:.4f}\n")
    buf.write("\n--- Input Text (first 500 chars) ---\n")
    buf.write(text[:500] + ("..." if len(text) > 500 else "") + "\n")
    return buf.getvalue().encode()


def generate_comparison_report(text, rows, ai_votes, human_votes,
                               llm_summary: str = "") -> bytes:
    from datetime import datetime
    total = ai_votes + human_votes
    if ai_votes == human_votes:
        consensus = "Split decision"
    else:
        consensus = "AI-Generated" if ai_votes > human_votes else "Human-Written"
    buf = io.StringIO()
    buf.write("DetectIQ — Multi-Model Comparison Report\n")
    buf.write("=" * 55 + "\n")
    buf.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
    buf.write(f"Consensus     : {consensus}\n")
    buf.write(f"Vote split    : {human_votes} Human / {ai_votes} AI "
              f"({max(ai_votes, human_votes)}/{total} agree)\n\n")
    buf.write("Per-Model Results\n" + "-"*46 + "\n")
    buf.write(f"  {'Model':<16}{'Prediction':<16}{'Confidence':<12}\n")
    for r in rows:
        buf.write(f"  {r['Model']:<16}{r['Prediction']:<16}{r['Confidence']:<12}\n")
    if llm_summary:
        buf.write("\nLLM Summary — Model Consensus\n" + "-"*46 + "\n")
        buf.write(llm_summary + "\n")
    buf.write("\n--- Input Text (first 500 chars) ---\n")
    buf.write(text[:500] + ("..." if len(text) > 500 else "") + "\n")
    return buf.getvalue().encode()


MODEL_DESCS = {
    "SVM": "Linear boundary on TF-IDF space.",
    "Decision Tree": "Interpretable rule-based tree.",
    "AdaBoost": "Ensemble of weak DT classifiers.",
    "FNN": "Dense network on TF-IDF features.",
    "LSTM": "Recurrent model over word sequences.",
    "CNN": "Multi-filter conv over word embeddings.",
}

MODEL_TYPE = {
    "SVM": "Traditional ML", "Decision Tree": "Traditional ML",
    "AdaBoost": "Traditional ML", "FNN": "Deep Learning",
    "LSTM": "Deep Learning", "CNN": "Deep Learning",
}


# Hero verdict card — clean result + confidence meter
def render_verdict(pred: dict, model_name: str):
    is_ai = pred["label"] == 1
    cls   = "v-ai" if is_ai else "v-human"
    icon  = "🤖" if is_ai else "✍️"
    conf  = pred["confidence"] * 100
    strength = ("Very high" if conf >= 90 else "High" if conf >= 75
                else "Moderate" if conf >= 60 else "Low")
    plain = ("This text most likely came from an AI system."
             if is_ai else "This text reads as written by a person.")
    st.markdown(
        f'<div class="verdict-card {cls}">'
          f'<div class="verdict-top">'
            f'<div class="verdict-label">{icon}&nbsp;{pred["label_str"]}</div>'
            f'<div class="model-pill">Analyzed by · {model_name}</div>'
          f'</div>'
          f'<div class="conf-wrap">'
            f'<div class="conf-row"><span>Model confidence ({strength})</span>'
            f'<span class="conf-pct">{conf:.1f}%</span></div>'
            f'<div class="conf-track"><div class="conf-fill" '
            f'style="width:{conf:.1f}%"></div></div>'
          f'</div>'
          f'<div class="verdict-foot">{plain}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# LLM output panel with a titled header
def render_llm_panel(title: str, model_line: str, body: str):
    st.markdown(
        f'<div class="llm-box"><div class="llm-head">{title}</div>'
        f'<div class="llm-sub">{model_line}</div>{body}</div>',
        unsafe_allow_html=True,
    )


# Compact per-model verdict card used in the side-by-side comparison
def mini_verdict_html(model_name: str, p: dict) -> str:
    is_ai = p["label"] == 1
    color = "#c92a2a" if is_ai else "#2b8a3e"
    icon  = "🤖" if is_ai else "✍️"
    return (
        f'<div class="mini-card" style="border-top:5px solid {color}">'
        f'<div class="mini-model">{model_name}</div>'
        f'<div class="mini-type">{MODEL_TYPE.get(model_name, "")}</div>'
        f'<div class="mini-verdict" style="color:{color}">{icon} {p["label_str"]}</div>'
        f'<div class="mini-conf">{p["confidence"]*100:.0f}% confidence</div>'
        f'</div>'
    )


def render_consensus(ai_votes: int, human_votes: int):
    total = ai_votes + human_votes
    if ai_votes == human_votes:
        main = "Split decision"
        bg   = "linear-gradient(135deg,#868e96,#adb5bd)"
    elif ai_votes > human_votes:
        main = "AI-Generated"
        bg   = "linear-gradient(135deg,#c92a2a,#ff6b6b)"
    else:
        main = "Human-Written"
        bg   = "linear-gradient(135deg,#2b8a3e,#51cf66)"
    agree = max(ai_votes, human_votes)
    st.markdown(
        f'<div class="consensus-card" style="background:{bg}">'
        f'<div class="consensus-head">Consensus of {total} models</div>'
        f'<div class="consensus-main">{main}</div>'
        f'<div style="font-size:.85rem;opacity:.92;margin-top:2px">'
        f'{agree}/{total} models agree · {human_votes} Human, {ai_votes} AI</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


NAV_ITEMS = [("Predict", "predict"), ("Compare", "compare"),
             ("Benchmark", "benchmark"), ("About", "about")]


def read_uploaded(file) -> str:
    if file is None:
        return ""
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    if file.name.endswith(".docx"):
        return extract_text_from_docx(file)
    return file.read().decode("utf-8", errors="ignore")


def reset_page_state():
    for k in ("predict_result", "compare_result", "auto_run",
              "pp_text", "cp_text", "pp_file", "cp_file",
              "pp_seed", "cp_seed", "pp_user_typed", "cp_user_typed"):
        st.session_state.pop(k, None)


def render_top_nav(active: str):
    with st.container(key="topnav"):
        cols = st.columns([2.6, 4.4, 1, 1, 1.3, 1])
        with cols[0]:
            if st.button("DetectIQ", key="nav_brand"):
                reset_page_state()
                st.session_state.page = "home"
                st.rerun()
        for col, (label, pg) in zip(cols[2:], NAV_ITEMS):
            with col:
                if st.button(label, key=f"nav_{pg}",
                             type="primary" if active == pg else "secondary"):
                    reset_page_state()
                    st.session_state.page = pg
                    st.rerun()


def main():
    ml_models, ml_errors = load_ml_models()
    llm_errors = []

    available_models = [m for m in
        ["SVM", "Decision Tree", "AdaBoost", "FNN", "LSTM", "CNN"]
        if m in ml_models]

    st.session_state.setdefault("page", "home")

    def llm_caption():
        st.caption("🟡 LLMs enabled · models load when analysis runs")

    def run_predict(text, model_name, use_llm):
        pred = predict(text, model_name, ml_models)
        if "error" in pred:
            return {"error": pred["error"]}
        return {"text": text, "model_name": model_name, "use_llm": use_llm,
                "pred": pred, "stats": compute_text_stats(text),
                "explanation": "(LLM disabled)", "style": "(LLM disabled)",
                "llm_done": False}

    def render_predict_result(r):
        pred = r["pred"]
        render_verdict(pred, r["model_name"])

        labels = ["Overview"]
        if r["use_llm"]:
            labels.append("LLM Analysis")
        labels += ["Model Insight", "Linguistic Signals"]
        tabs = dict(zip(labels, st.tabs(labels)))

        with tabs["Overview"]:
            stats = render_text_stats(r["text"])
        with tabs["Model Insight"]:
            render_ml_explanation(r["model_name"], r["text"], pred, ml_models)
        with tabs["Linguistic Signals"]:
            ling_df = pd.DataFrame({
                "Feature": LING_FEATURE_NAMES,
                "Value":   [f"{v:.4f}" for v in pred["ling_raw"]],
                "Z-score": [f"{v:+.3f}" for v in pred["ling_scaled"]],
            })
            st.dataframe(ling_df, use_container_width=True, height=560)
        if r["use_llm"]:
            with tabs["LLM Analysis"]:
                c1, c2 = st.columns(2)
                with c1:
                    with st.container(border=True):
                        st.markdown(
                            '<div class="llm-head">LLM 1 — Prediction Explanation</div>'
                            '<div class="llm-sub">Qwen2.5-1.5B-Instruct explains why '
                            'the ML model made this call</div>',
                            unsafe_allow_html=True)
                        if r["llm_done"]:
                            st.markdown(r["explanation"])
                        else:
                            prompt = build_explain_prompt(
                                r["text"], pred["label_str"], pred["confidence"],
                                get_top_features(pred))
                            try:
                                llm1 = get_llm1()
                                r["explanation"] = stream_with_spinner(
                                    stream_llm(llm1, prompt, 300),
                                    "LLM 1 is thinking…")
                            except Exception as e:
                                r["explanation"] = f"LLM 1 unavailable: {e}"
                                st.info(r["explanation"])
                with c2:
                    with st.container(border=True):
                        st.markdown(
                            '<div class="llm-head">LLM 2 — Style Analysis</div>'
                            '<div class="llm-sub">TinyLlama-1.1B-Chat on sentence '
                            'uniformity, hedging, vocabulary</div>',
                            unsafe_allow_html=True)
                        if r["llm_done"]:
                            st.markdown(r["style"])
                        else:
                            prompt = build_style_prompt(r["text"], r["stats"])
                            try:
                                llm2 = get_llm2()
                                r["style"] = stream_with_spinner(
                                    stream_llm(llm2, prompt, 280),
                                    "LLM 2 is thinking…")
                            except Exception as e:
                                r["style"] = f"LLM 2 unavailable: {e}"
                                st.info(r["style"])
                r["llm_done"] = True

        report = generate_report(r["text"], pred, r["model_name"], stats,
                                 r["explanation"], r["style"])
        st.download_button("⬇ Download Full Report", data=report,
                           file_name="detection_report.txt", mime="text/plain")

    def run_compare(text, use_llm):
        rows, preds = [], {}
        for mn in available_models:
            p = predict(text, mn, ml_models)
            if "error" not in p:
                preds[mn] = p
                rows.append({"Model": mn, "Type": MODEL_TYPE.get(mn, ""),
                             "Prediction": p["label_str"],
                             "Confidence": f"{p['confidence']*100:.1f}%"})
        ai_votes = sum(1 for r in rows if r["Prediction"] == "AI-Generated")
        human_votes = len(rows) - ai_votes

        return {"text": text, "rows": rows, "preds": preds, "use_llm": use_llm,
                "ai_votes": ai_votes, "human_votes": human_votes,
                "stats": compute_text_stats(text),
                "summary": "", "summary_done": False,
                "style": "", "style_done": False}

    def render_compare_result(r):
        rows, preds = r["rows"], r["preds"]
        if not rows:
            st.error("No models produced a prediction.")
            return
        render_consensus(r["ai_votes"], r["human_votes"])

        model_list = list(preds.keys())
        for i in range(0, len(model_list), 3):
            cols = st.columns(3)
            for col, mn in zip(cols, model_list[i:i+3]):
                col.markdown(mini_verdict_html(mn, preds[mn]), unsafe_allow_html=True)
        st.markdown("")

        fig, ax = plt.subplots(figsize=(8, 3))
        names  = list(preds.keys())
        confs  = [preds[m]["confidence"]*100 for m in names]
        colors = ["#c92a2a" if preds[m]["label"] == 1 else "#2b8a3e" for m in names]
        ax.bar(names, confs, color=colors)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Confidence (%)")
        ax.set_title("Per-model confidence  (red = AI, green = Human)")
        for j, v in enumerate(confs):
            ax.text(j, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if r["use_llm"]:
            st.markdown("")
            lc1, lc2 = st.columns(2)
            with lc1:
                with st.container(border=True):
                    st.markdown(
                        '<div class="llm-head">LLM 1 — Model Consensus</div>'
                        '<div class="llm-sub">Qwen2.5-1.5B-Instruct interprets where '
                        'the six models agree and disagree</div>',
                        unsafe_allow_html=True)
                    if r["summary_done"]:
                        st.markdown(r["summary"])
                    else:
                        prompt = build_consensus_prompt(
                            r["text"], rows, r["ai_votes"], r["human_votes"])
                        try:
                            llm1 = get_llm1()
                            r["summary"] = stream_with_spinner(
                                stream_llm(llm1, prompt, 300),
                                "LLM 1 is summarizing the six models…")
                            r["summary_done"] = True
                        except Exception as e:
                            r["summary"] = f"LLM 1 unavailable: {e}"
                            st.info(r["summary"])
            with lc2:
                with st.container(border=True):
                    st.markdown(
                        '<div class="llm-head">LLM 2 — Style Analysis</div>'
                        '<div class="llm-sub">TinyLlama-1.1B-Chat reads the text for '
                        'authorship signals — sentence uniformity, hedging, '
                        'vocabulary</div>', unsafe_allow_html=True)
                    if r["style_done"]:
                        st.markdown(r["style"])
                    else:
                        prompt = build_style_prompt(r["text"], r["stats"])
                        try:
                            llm2 = get_llm2()
                            r["style"] = stream_with_spinner(
                                stream_llm(llm2, prompt, 280),
                                "LLM 2 is analyzing writing style…")
                            r["style_done"] = True
                        except Exception as e:
                            r["style"] = f"LLM 2 unavailable: {e}"
                            st.info(r["style"])
        elif r["use_llm"]:
            st.info("LLMs unavailable — summaries skipped.")

        cmp_report = generate_comparison_report(
            r["text"], rows, r["ai_votes"], r["human_votes"], r["summary"])
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Comparison Report (TXT)", cmp_report,
                           "detectiq_comparison_report.txt", "text/plain",
                           use_container_width=True)
        d2.download_button("⬇ Results (CSV)",
                           pd.DataFrame(rows).to_csv(index=False).encode(),
                           "detectiq_comparison.csv", "text/csv",
                           use_container_width=True)

    def page_home():
        left, right = st.columns([1, 1.2], gap="large")

        with left:
            st.markdown(
                '<div class="hero-text">'
                '<div class="hero-eyebrow">AI vs Human</div>'
                '<h1>Detect <span>AI-generated</span> text with confidence.</h1>'
                '<p>DetectIQ classifies any passage as AI-written or human-written '
                'using six trained machine learning &amp; deep learning models, compares them '
                'side-by-side, and lets two LLMs explain <em>why</em>.</p>'
                '<div class="try-it">Try it out '
                '<span class="arrow">➜</span></div>'
                '<div class="feat-row">'
                '<div class="feat"><b>6</b> ML / DL models</div>'
                '<div class="feat"><b>2</b> explainer LLMs</div>'
                '<div class="feat"><b>15</b> linguistic signals</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )

        with right:
            with st.container(border=True, key="trybox"):
                t_ai, t_cmp = st.tabs(["AI vs Human Detector", "Model Comparison"])

                with t_ai:
                    text = st.text_area(
                        "input", height=210, key="home_ai_text",
                        label_visibility="collapsed",
                        placeholder="Paste your text here to check if it's "
                                    "AI-generated or human-written…")
                    up = st.file_uploader("📎 Upload PDF / Word / TXT",
                                          type=["pdf", "docx", "txt"], key="home_ai_file")
                    o1, o2 = st.columns([1.5, 1.2])
                    with o1:
                        model = st.selectbox("Model", available_models, key="home_ai_model")
                    with o2:
                        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                        use_llm = st.toggle("Enable LLM analysis", value=True,
                                            key="home_ai_llm")
                    llm_caption()
                    _, bcol = st.columns([2.4, 1])
                    with bcol:
                        go = st.button("Predict  ➜", type="primary",
                                       use_container_width=True, key="home_predict_btn")
                    if go:
                        src = read_uploaded(up) or text
                        if src.strip():
                            st.session_state.shared_text = src
                            st.session_state.sel_model = model
                            st.session_state.use_llm = use_llm
                            st.session_state.page = "predict"
                            st.session_state.auto_run = True
                            st.rerun()
                        else:
                            st.warning("Enter or upload some text first.")

                with t_cmp:
                    ctext = st.text_area(
                        "input", height=210, key="home_cmp_text",
                        label_visibility="collapsed",
                        placeholder="Paste text to run through all six models at once…")
                    cup = st.file_uploader("📎 Upload PDF / Word / TXT",
                                           type=["pdf", "docx", "txt"], key="home_cmp_file")
                    cuse_llm = st.toggle("Enable LLM analysis", value=True,
                                         key="home_cmp_llm")
                    llm_caption()
                    _, bcol = st.columns([2.4, 1])
                    with bcol:
                        cgo = st.button("Compare  ➜", type="primary",
                                        use_container_width=True, key="home_compare_btn")
                    if cgo:
                        src = read_uploaded(cup) or ctext
                        if src.strip():
                            st.session_state.shared_text = src
                            st.session_state.use_llm = cuse_llm
                            st.session_state.page = "compare"
                            st.session_state.auto_run = True
                            st.rerun()
                        else:
                            st.warning("Enter or upload some text first.")

    def page_predict():
        st.subheader("AI vs Human Detector")
        st.caption("Paste text below, pick a model, and run a single-model prediction.")

        auto = st.session_state.pop("auto_run", False)
        if auto:
            st.session_state["pp_seed"]       = st.session_state.get("shared_text", "")
            st.session_state["pp_user_typed"] = False
            st.session_state["pp_model"]      = st.session_state.get("sel_model", available_models[0])
            st.session_state["pp_llm"]        = st.session_state.get("use_llm", True)
        if "pp_seed" in st.session_state and not st.session_state.get("pp_user_typed"):
            st.session_state["pp_text"] = st.session_state["pp_seed"]

        with st.container(border=True, key="pagebox"):
            text = st.text_area("input", height=320, key="pp_text",
                                label_visibility="collapsed",
                                on_change=lambda: st.session_state.update(pp_user_typed=True),
                                placeholder="Paste any text — essay, article, email…")
            up = st.file_uploader("📎 Upload PDF / Word / TXT",
                                  type=["pdf", "docx", "txt"], key="pp_file")
            o1, o2 = st.columns([1.5, 1.2])
            with o1:
                model = st.selectbox("Model", available_models, key="pp_model")
            with o2:
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                st.session_state.setdefault("pp_llm", True)
                use_llm = st.toggle("Enable LLM analysis", key="pp_llm")
            st.caption(MODEL_DESCS.get(model, ""))
            llm_caption()
            _, bcol = st.columns([2.4, 1])
            with bcol:
                clicked = st.button("Predict  ➜", type="primary",
                                    use_container_width=True, key="pp_btn")

        src = read_uploaded(up) or text
        if (clicked or auto) and src.strip():
            with st.spinner("Running ML classifier…"):
                st.session_state.predict_result = run_predict(src, model, use_llm)

        res = st.session_state.get("predict_result")
        if res and "error" in res:
            st.error(res["error"])
        elif res:
            st.divider()
            render_predict_result(res)

    def page_compare():
        st.subheader("Model Comparison")
        st.caption("Every model classifies the same text — live predictions on your "
                   "input, not stored training scores.")

        auto = st.session_state.pop("auto_run", False)
        if auto:
            st.session_state["cp_seed"]       = st.session_state.get("shared_text", "")
            st.session_state["cp_user_typed"] = False
            st.session_state["cp_llm"]        = st.session_state.get("use_llm", True)
        if "cp_seed" in st.session_state and not st.session_state.get("cp_user_typed"):
            st.session_state["cp_text"] = st.session_state["cp_seed"]

        with st.container(border=True, key="pagebox"):
            text = st.text_area("input", height=320, key="cp_text",
                                label_visibility="collapsed",
                                on_change=lambda: st.session_state.update(cp_user_typed=True),
                                placeholder="Paste text to run through all six models…")
            up = st.file_uploader("📎 Upload PDF / Word / TXT",
                                  type=["pdf", "docx", "txt"], key="cp_file")
            st.session_state.setdefault("cp_llm", True)
            use_llm = st.toggle("Enable LLM analysis (consensus + style)", key="cp_llm")
            llm_caption()
            _, bcol = st.columns([2.4, 1])
            with bcol:
                clicked = st.button("Compare  ➜", type="primary",
                                    use_container_width=True, key="cp_btn")

        src = read_uploaded(up) or text
        if (clicked or auto) and src.strip():
            with st.spinner("Running all six models…"):
                st.session_state.compare_result = run_compare(src, use_llm)

        res = st.session_state.get("compare_result")
        if res:
            st.divider()
            render_compare_result(res)

    def page_benchmark():
        st.subheader("Benchmark Performance")
        st.caption("Fixed scores from the held-out test set during training — how "
                   "reliable each model is overall, independent of any text you paste.")
        if "results" not in ml_models:
            st.info("model_results.json not found — train-time metrics unavailable.")
            return

        res = ml_models["results"]
        metrics = ["acc", "prec", "rec", "f1", "roc_auc"]
        labels  = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
        df_cmp = pd.DataFrame([
            {"Model": k, **{lbl: round(v[m], 4) for m, lbl in zip(metrics, labels)}}
            for k, v in res.items()
        ]).sort_values("F1", ascending=False).reset_index(drop=True)

        styled = df_cmp.style.highlight_max(subset=labels, color="#d3f9d8")
        st.dataframe(styled, use_container_width=True, hide_index=True)

        bench_tabs = st.tabs(["Metric bars", "ROC curves"])
        with bench_tabs[0]:
            fig, ax = plt.subplots(figsize=(12, 5))
            x = np.arange(len(df_cmp)); w = 0.15
            for i, (lbl, m) in enumerate(zip(labels, metrics)):
                ax.bar(x + i*w, df_cmp[lbl], w, label=lbl, color=plt.cm.Set2.colors[i])
            ax.set_xticks(x + w*2)
            ax.set_xticklabels(df_cmp["Model"], rotation=15)
            ax.set_ylim(0.5, 1.05)
            ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
            ax.set_title("All Models — Performance Metrics")
            ax.set_ylabel("Score")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with bench_tabs[1]:
            if all("fpr" in r for r in res.values()):
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                for (name, r), color in zip(res.items(), plt.cm.tab10.colors):
                    ax2.plot(r["fpr"], r["tpr"], lw=2, color=color,
                             label=f"{name} (AUC={r['roc_auc']:.3f})")
                ax2.plot([0, 1], [0, 1], "k--", lw=1)
                ax2.set_xlabel("False Positive Rate")
                ax2.set_ylabel("True Positive Rate")
                ax2.set_title("ROC Curves")
                ax2.legend(loc="lower right", fontsize=9)
                ax2.grid(alpha=0.3)
                fig2.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)
            else:
                st.info("ROC curve data not present in model_results.json.")

    def page_about():
        st.markdown("""
## About DetectIQ

**DetectIQ — AI vs Human Text Detector with LLM Integration**

DetectIQ classifies any text as AI-generated or human-written using six trained
ML/DL models, lets you compare all of them side-by-side on the same text, and uses
two LLMs to explain *why* a piece of text reads the way it does.

### ML Models
| Model | Feature | Type |
|-------|---------|------|
| SVM | TF-IDF (10K, bigrams) | Traditional ML |
| Decision Tree | TF-IDF (10K, bigrams) | Traditional ML |
| AdaBoost | TF-IDF (10K, bigrams) | Traditional ML |
| FNN | TF-IDF dense | Deep Learning |
| LSTM | Token sequences (maxlen=300) | Deep Learning |
| CNN | Token sequences (maxlen=300) | Deep Learning |

### LLMs
| LLM | Model | Role |
|-----|-------|------|
| LLM 1 | Qwen2.5-1.5B-Instruct | Explains *why* the ML model made its prediction |
| LLM 2 | TinyLlama-1.1B-Chat | Structured stylistic analysis of sentence uniformity, hedging, vocabulary |

### Feature Engineering
- **TF-IDF** — 10K features, unigrams+bigrams, sublinear TF
- **GloVe / LSA** — 100-dimensional dense embeddings
- **Linguistic** — 15 stylometric signals including Flesch score, hapax legomena ratio, vocab richness

### Dataset
- 8,176 samples: 4,088 human-written, 4,088 AI-generated (perfectly balanced)

### Deployment
Deployed on [Hugging Face Spaces](https://huggingface.co/spaces/thelege9d/DetectIQ)

### Tech Stack
`scikit-learn` · `TensorFlow/Keras` · `Transformers (HuggingFace)` · `NLTK` · `Streamlit`
        """)
        all_errors = ml_errors + llm_errors
        if all_errors:
            with st.expander("⚠ Load warnings"):
                for e in all_errors:
                    st.warning(e)

    render_top_nav(st.session_state.page)

    if not available_models:
        st.error("No ML models loaded.")
        st.stop()

    router = {
        "home": page_home, "predict": page_predict, "compare": page_compare,
        "benchmark": page_benchmark, "about": page_about,
    }
    router.get(st.session_state.page, page_home)()


if __name__ == "__main__":
    main()
