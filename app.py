# app.py
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from openai import OpenAI

# Initialize OpenAI client
OPENAI_API_KEY = st.secrets["api_keys"]["openai"]
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Translation functions ---
@st.cache_resource
def load_model(src_lang="en", tgt_lang="hi"):
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
        # For unsupported language pairs, raise error to trigger OpenAI fallback
        raise ValueError(f"No local model available for {src_lang} to {tgt_lang} translation")

    model_name = model_mapping[model_key]
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model


def translate_text(text, src_lang="en", tgt_lang="hi"):
    try:
        tokenizer, model = load_model(src_lang, tgt_lang)
        inputs = tokenizer([text], return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        return translated_text
    except (ValueError, Exception) as e:
        # Fallback to OpenAI if model not available
        # st.warning(f"Using OpenAI fallback for translation: {str(e)}")
        return openai_translate_text(text, src_lang, tgt_lang)


def openai_translate_text(text, src_lang="en", tgt_lang="hi"):
    """Fallback translation using OpenAI"""
    if not OPENAI_API_KEY:
        return "Please set OPENAI_API_KEY to use translation"

    prompt = f"Translate the following text from {src_lang} to {tgt_lang}: {text}"
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Translation error: {str(e)}"


def explain_context(original_text, translated_text, src_lang="en", tgt_lang="hi"):
    if not OPENAI_API_KEY:
        return "Please set OpenAI API key to get context explanations"

    prompt = f"""
    Original text ({src_lang}): {original_text}
    Translated text ({tgt_lang}): {translated_text}

    Please explain:
    1. Why this translation was chosen.
    2. If the original text contains idioms, metaphors, or slang, clarify their cultural meaning.
    3. Suggest alternative phrasings if available.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Context explanation error: {str(e)}"


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