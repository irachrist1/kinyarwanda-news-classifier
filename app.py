"""Interactive Kinyarwanda news topic classifier."""

import json
from pathlib import Path

import joblib
import streamlit as st


ROOT = Path(__file__).resolve().parent
ENGLISH_DISPLAY = {"politic": "Politics", "sport": "Sports", "relationship": "Relationships"}


@st.cache_resource
def load_resources():
    model = joblib.load(ROOT / "artifacts" / "final.joblib")
    labels = json.loads((ROOT / "artifacts" / "labels.json").read_text())
    return model, labels


st.set_page_config(page_title="Kinyarwanda News Topics", layout="centered")
st.title("Kinyarwanda news topics")
st.write("Paste a Kinyarwanda news headline or article to see its predicted topic.")

with st.form("classify"):
    article = st.text_area("Kinyarwanda news text", height=220, placeholder="Andika cyangwa shyiramo inkuru hano...")
    submitted = st.form_submit_button("Classify topic", type="primary")

if submitted:
    if not article.strip():
        st.warning("Enter some news text first.")
    else:
        model, labels = load_resources()
        prediction = int(model.predict([article])[0])
        name = labels[str(prediction)]
        st.subheader(name["kinyarwanda"].capitalize())
        english = ENGLISH_DISPLAY.get(name["english"], name["english"].capitalize())
        st.write(f"English topic: {english}")

st.caption("Trained on Kinyarwanda news articles. Predictions for short messages or other types of text may be less reliable.")
st.caption("Data: [KINNEWS corpus](https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus). [Model evaluation](https://github.com/irachrist1/kinyarwanda-news-classifier/blob/main/results/test_metrics.json).")
