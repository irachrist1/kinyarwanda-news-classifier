"""Interactive Kinyarwanda news topic classifier."""

import streamlit as st

from src.inference import classify


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
        result = classify(article)
        st.subheader(result["kinyarwanda"])
        st.write(f"English topic: {result['english']}")

st.caption("Trained on Kinyarwanda news articles. Predictions for short messages or other types of text may be less reliable.")
st.caption("Data: [KINNEWS corpus](https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus). [Model evaluation](https://github.com/irachrist1/kinyarwanda-news-classifier/blob/main/results/test_metrics.json).")
