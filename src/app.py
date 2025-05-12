import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Chargement du modèle fine-tuné
@st.cache_resource
def load_model():
    model_path = "./tweet-model"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# Génération d’un tweet à partir du sentiment sélectionné
def generate_tweet_finetuned(sentiment="positive", max_length=50):
    prompt = f"<{sentiment}>"
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

if st.button("Générer un tweet"):
    with st.spinner("Génération en cours..."):
        tweet = generate_tweet_finetuned(sentiment)
    st.success("Tweet généré :")
    st.write(tweet)

