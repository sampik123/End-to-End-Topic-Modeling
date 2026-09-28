"""Precompute LDA topic-model artifacts for the Streamlit app.

Run this once (or whenever the data/model changes):
    venv\\Scripts\\python.exe prepare_artifacts.py

It reproduces the same cleaning + LDA pipeline as Topic_Modelling.ipynb, then
saves everything the Streamlit app needs (model, dictionary, per-document
topic assignments, top words per topic, representative headlines, and a
pyLDAvis HTML map) into ./artifacts/, so the app never has to retrain.
"""
import json
import re
from pathlib import Path

import contractions
import gensim
import gensim.corpora as corpora
import nltk
import numpy as np
import pandas as pd
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "final_data_topic_modelling.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
NUM_TOPICS = 20
TOP_N_WORDS = 15
TOP_N_DOCS_PER_TOPIC = 8

for pkg in ("stopwords", "punkt", "punkt_tab", "wordnet"):
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def fix_mojibake(text: str) -> str:
    """Recover UTF-8 text that was mis-decoded as Latin-1 (e.g. curly quotes -> 'â€™')."""
    try:
        return text.encode("latin1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def clean_to_tokens(text: str) -> list[str]:
    text = " ".join(contractions.fix(w) for w in str(text).split())
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return tokens


def main():
    print("Loading raw data...")
    raw = pd.read_csv(DATA_PATH, encoding="latin1")
    df = raw[["Title", "Content"]].dropna().drop_duplicates().reset_index(drop=True)
    df["Title"] = df["Title"].apply(fix_mojibake)
    print(f"{len(df)} articles after dropping empty/duplicate rows.")

    print("Cleaning text (this is the slow step, be patient)...")
    df["tokens"] = df["Content"].apply(clean_to_tokens)
    df["word_count"] = df["tokens"].apply(len)
    df = df[df["word_count"] > 5].reset_index(drop=True)
    print(f"{len(df)} articles after removing near-empty documents.")

    print("Building dictionary + corpus...")
    id2word = corpora.Dictionary(df["tokens"])
    id2word.filter_extremes(no_below=5, no_above=0.5)
    corpus = [id2word.doc2bow(tokens) for tokens in df["tokens"]]

    print(f"Training LDA with {NUM_TOPICS} topics (passes=10)...")
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=NUM_TOPICS,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha="auto",
        per_word_topics=True,
    )

    print("Scoring model...")
    perplexity = lda_model.log_perplexity(corpus)
    coherence_model = CoherenceModel(
        model=lda_model, texts=df["tokens"], dictionary=id2word, coherence="c_v"
    )
    with np.errstate(invalid="ignore"):
        coherence = coherence_model.get_coherence()
    print(f"Perplexity: {perplexity:.3f}  Coherence: {coherence:.3f}")

    print("Assigning topics to documents...")
    dominant_topic, dominant_prob = [], []
    for bow in corpus:
        dist = lda_model.get_document_topics(bow, minimum_probability=0.0)
        top_tid, top_p = max(dist, key=lambda x: x[1])
        dominant_topic.append(top_tid)
        dominant_prob.append(float(top_p))
    df["dominant_topic"] = dominant_topic
    df["topic_prob"] = dominant_prob

    print("Extracting top words per topic...")
    topic_words = {}
    for tid in range(NUM_TOPICS):
        words = lda_model.show_topic(tid, topn=TOP_N_WORDS)
        topic_words[str(tid)] = [{"word": w, "weight": float(wt)} for w, wt in words]

    topic_labels = {
        str(tid): " / ".join(w["word"] for w in topic_words[str(tid)][:3])
        for tid in range(NUM_TOPICS)
    }

    print("Selecting representative headlines per topic...")
    representative = {}
    for tid in range(NUM_TOPICS):
        sub = df[df["dominant_topic"] == tid].sort_values("topic_prob", ascending=False)
        representative[str(tid)] = sub[["Title", "topic_prob"]].head(TOP_N_DOCS_PER_TOPIC).to_dict(
            "records"
        )

    print("Saving artifacts...")
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    lda_model.save(str(ARTIFACTS_DIR / "lda_model.gensim"))
    id2word.save(str(ARTIFACTS_DIR / "dictionary.gensim"))

    df[["Title", "word_count", "dominant_topic", "topic_prob"]].to_pickle(
        ARTIFACTS_DIR / "doc_topics.pkl"
    )

    with open(ARTIFACTS_DIR / "topic_words.json", "w", encoding="utf-8") as f:
        json.dump(topic_words, f, indent=2)
    with open(ARTIFACTS_DIR / "topic_labels.json", "w", encoding="utf-8") as f:
        json.dump(topic_labels, f, indent=2)
    with open(ARTIFACTS_DIR / "representative_docs.json", "w", encoding="utf-8") as f:
        json.dump(representative, f, indent=2)
    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "perplexity": perplexity,
                "coherence": coherence,
                "num_docs": len(df),
                "num_topics": NUM_TOPICS,
            },
            f,
            indent=2,
        )

    print("Rendering pyLDAvis HTML map...")
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis

    vis = gensimvis.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(vis, str(ARTIFACTS_DIR / "pyldavis.html"))

    print("Done. Artifacts saved to", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
