# translator.py
from transformers import MarianMTModel, MarianTokenizer
import streamlit as st
from openai import OpenAI

OPENAI_API_KEY = st.secrets["api_keys"]["openai"]
client = OpenAI(api_key=OPENAI_API_KEY)


def load_model(src_lang="en", tgt_lang="hi"):
    """
    Load MarianMT model for translation.
    Example: English (en) → Hindi (hi)
    """
    # Use only verified available models from Helsinki-NLP
    model_mapping = {
        ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
        ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
        ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
        ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
        ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
        ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
        ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
        ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en"
    }

    model_key = (src_lang, tgt_lang)
    if model_key not in model_mapping:
        raise ValueError(f"No model available for {src_lang} to {tgt_lang} translation")

    model_name = model_mapping[model_key]
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model


def translate_text(text, src_lang="en", tgt_lang="hi"):
    """
    Translate text from source to target language
    """
    tokenizer, model = load_model(src_lang, tgt_lang)
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text


def openai_translate_text(text, src_lang="en", tgt_lang="hi", model="gpt-3.5-turbo"):
    """
    Translate text using OpenAI's GPT model via API.
    """
    prompt = f"""
    Translate the following text from {src_lang} to {tgt_lang}:
    {text}
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()


# 🚀 Example usage
if __name__ == "__main__":
    source_text = "Good morning! How are you?"
    print("--- HuggingFace MarianMT ---")
    output_hf = translate_text(source_text, src_lang="en", tgt_lang="fr")
    print(f"Input: {source_text}")
    print(f"Translated: {output_hf}")

    print("\n--- OpenAI GPT ---")
    output_openai = openai_translate_text(source_text, src_lang="en", tgt_lang="fr")
    print(f"Input: {source_text}")
    print(f"Translated: {output_openai}")