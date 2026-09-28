"""Streamlit demo for the news-article topic model.

Run with:
    venv\\Scripts\\streamlit run app.py

Loads precomputed artifacts from ./artifacts/ (see prepare_artifacts.py) -
no model training happens here, so the app starts and responds instantly.
"""
import json
from pathlib import Path

import gensim
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

st.set_page_config(page_title="News Topic Explorer", page_icon="📰", layout="wide")


@st.cache_resource
def load_model():
    return gensim.models.LdaModel.load(str(ARTIFACTS_DIR / "lda_model.gensim"))


@st.cache_data
def load_json(name: str):
    with open(ARTIFACTS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_doc_topics():
    return pd.read_pickle(ARTIFACTS_DIR / "doc_topics.pkl")


if not ARTIFACTS_DIR.exists():
    st.error(
        "No artifacts found. Run `venv\\Scripts\\python.exe prepare_artifacts.py` "
        "first to generate the topic model."
    )
    st.stop()

topic_words = load_json("topic_words.json")
topic_labels = load_json("topic_labels.json")
representative_docs = load_json("representative_docs.json")
metrics = load_json("metrics.json")
doc_topics = load_doc_topics()

topic_ids = sorted(topic_words.keys(), key=int)
topic_display = {tid: f"Topic {int(tid) + 1}: {topic_labels[tid]}" for tid in topic_ids}

st.title("📰 News Article Topic Explorer")
st.caption(
    f"LDA topic model over {metrics['num_docs']:,} news articles · "
    f"{metrics['num_topics']} topics · "
    f"perplexity {metrics['perplexity']:.3f} · coherence {metrics['coherence']:.3f}"
)

tab_overview, tab_topic, tab_map = st.tabs(["Overview", "Topic Explorer", "Topic Map"])

with tab_overview:
    st.subheader("Topic sizes across the corpus")
    counts = doc_topics["dominant_topic"].value_counts().sort_index()
    size_df = pd.DataFrame(
        {
            "Topic": [topic_display[str(t)] for t in counts.index],
            "Articles": counts.values,
        }
    ).sort_values("Articles", ascending=True)
    fig = px.bar(size_df, x="Articles", y="Topic", orientation="h", height=650)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Articles modeled", f"{metrics['num_docs']:,}")
    col2.metric("Topics", metrics["num_topics"])
    col3.metric("Coherence (c_v)", f"{metrics['coherence']:.3f}")

with tab_topic:
    selected_label = st.selectbox(
        "Choose a topic", options=topic_ids, format_func=lambda t: topic_display[t]
    )

    col_words, col_docs = st.columns([1, 1.4])

    with col_words:
        st.markdown("**Top words**")
        words_df = pd.DataFrame(topic_words[selected_label])
        words_df = words_df.sort_values("weight", ascending=True)
        fig_words = px.bar(
            words_df, x="weight", y="word", orientation="h", height=450
        )
        fig_words.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_words, use_container_width=True)

    with col_docs:
        st.markdown("**Most representative headlines**")
        docs = representative_docs[selected_label]
        for d in docs:
            st.markdown(f"- {d['Title']}  \n  <span style='opacity:0.6'>match: {d['topic_prob']:.0%}</span>", unsafe_allow_html=True)

        n_in_topic = int((doc_topics["dominant_topic"] == int(selected_label)).sum())
        st.caption(f"{n_in_topic:,} articles have this as their dominant topic.")

with tab_map:
    st.subheader("Interactive topic map (pyLDAvis)")
    pyldavis_path = ARTIFACTS_DIR / "pyldavis.html"
    if pyldavis_path.exists():
        html = pyldavis_path.read_text(encoding="utf-8")
        st.components.v1.html(html, height=800, scrolling=True)
    else:
        st.info("pyLDAvis map not found — re-run prepare_artifacts.py to generate it.")
