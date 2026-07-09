import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os
import pandas as pd


MODEL_PATH = "./tweet-model"

# Chargement du modèle fine-tuné
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    return tokenizer, model

def load_seed_posts(uploaded_file=None):
    source = uploaded_file or os.getenv("XQUIK_EXPORT_CSV")
    if not source:
        return []

    df = pd.read_csv(source)
    text_column = next(
        (column for column in ["text", "tweet", "full_text"] if column in df.columns),
        None,
    )
    if text_column is None:
        st.warning("CSV must include a text, tweet, or full_text column.")
        return []

    return [
        str(value).strip()
        for value in df[text_column].dropna().tolist()
        if str(value).strip()
    ][:25]

if not os.path.isdir(MODEL_PATH):
    st.error("Run the notebook first to create ./tweet-model before launching the app.")
    st.stop()

tokenizer, model = load_model()

# Génération d’un tweet à partir du sentiment sélectionné
def generate_tweet_finetuned(sentiment="positive", max_length=50, seed_text=""):
    prompt = f"<{sentiment}> {seed_text[:120]}".strip()
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

# Interface Streamlit
st.title("🧠 Générateur de Tweets Financiers")
st.markdown("Ce générateur utilise un modèle GPT-2 fine-tuné pour produire des tweets financiers selon un sentiment donné.")

sentiment = st.selectbox("Choisissez un sentiment :", ["positive", "neutral", "negative"])
uploaded_file = st.file_uploader("Importer un export Xquik ou CSV de tweets", type="csv")
seed_posts = load_seed_posts(uploaded_file)
seed_text = st.selectbox("Tweet source", [""] + seed_posts) if seed_posts else ""

if st.button("Générer un tweet"):
    with st.spinner("Génération en cours..."):
        tweet = generate_tweet_finetuned(sentiment, seed_text=seed_text)
    st.success("Tweet généré :")
    st.write(tweet)
