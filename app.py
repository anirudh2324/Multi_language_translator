# app.py
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# --- Translation functions ---
@st.cache_resource
def load_model(src_lang="en", tgt_lang="hi"):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_text(text, src_lang="en", tgt_lang="hi"):
    tokenizer, model = load_model(src_lang, tgt_lang)
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

def explain_context(original_text, translated_text, src_lang="en", tgt_lang="hi"):
    prompt = f"""
    Original text ({src_lang}): {original_text}
    Translated text ({tgt_lang}): {translated_text}

    Please explain:
    1. Why this translation was chosen.
    2. If the original text contains idioms, metaphors, or slang, clarify their cultural meaning.
    3. Suggest alternative phrasings if available.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )

    return response.choices[0].message.content


# --- Streamlit UI ---
st.set_page_config(page_title="🌍 Multilingual Translator with Context", page_icon="🌐", layout="centered")

st.title("🌍 Multilingual Translator with Context")
st.write("Translate text into different languages and get cultural/linguistic explanations.")

# Define supported languages
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese (Simplified)": "zh",
    "Japanese": "ja",
    "Telugu": "te"
}

# User input
text_input = st.text_area("Enter text to translate", "Break a leg!")

col1, col2 = st.columns(2)
with col1:
    src_lang_name = st.selectbox("Source language", list(LANGUAGES.keys()), index=0)
    src_lang = LANGUAGES[src_lang_name]
with col2:
    tgt_lang_name = st.selectbox("Target language", list(LANGUAGES.keys()), index=1)
    tgt_lang = LANGUAGES[tgt_lang_name]

if st.button("Translate & Explain"):
    with st.spinner("Translating..."):
        translated = translate_text(text_input, src_lang, tgt_lang)
    st.success(f"**Translated Text ({tgt_lang}):** {translated}")

    with st.spinner("Explaining context..."):
        explanation = explain_context(text_input, translated, src_lang, tgt_lang)
    st.info(f"**Context Explanation:**\n\n{explanation}")
