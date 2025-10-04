# # app.py
import streamlit as st
from dotenv import load_dotenv
from transformers import MarianMTModel, MarianTokenizer
import google.generativeai as genai
import os


load_dotenv()


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
        st.warning(f"Using Gemini fallback for translation: {str(e)}")
        return openai_translate_text(text, src_lang, tgt_lang)


def openai_translate_text(text, src_lang="en", tgt_lang="hi"):
    """Fallback translation using OpenAI"""

    prompt = f"Translate the following exact text from {src_lang} to {tgt_lang}in one word: {text}"

    user_message_dict = {
        'role': 'user',
        'parts': [{"text": prompt}]
    }

    google_api_key = os.getenv("GOOGLE_API_KEY")  # Or st.secrets["GOOGLE_API_KEY"]

    if not google_api_key:
        st.error("❌ Google API key missing or not set. Please provide a valid API key.")
        st.error("If you need Context Explanation: use Gemini with an API key.")
        return None

    genai.configure(api_key=google_api_key)

    model = genai.GenerativeModel("gemini-2.0-flash-lite")

    try:
        response = model.generate_content(user_message_dict)

        return response.text
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None


def explain_context(original_text, translated_text, src_lang="en", tgt_lang="hi"):
    prompt = f"""
    Original text ({src_lang}): {original_text}
    Translated text ({tgt_lang}): {translated_text}

    Please explain:
    1. Why this translation was chosen.
    2. If the original text contains idioms, metaphors, or slang, clarify their cultural meaning.
    3. Suggest alternative phrasings if available.
    """

    # Get the Google API key from environment variables or Streamlit secrets
    # It's highly recommended to use Streamlit secrets for deployment
    google_api_key = os.getenv("GOOGLE_API_KEY") # Or st.secrets["GOOGLE_API_KEY"]

    if not google_api_key:
        st.error("❌ Google API key missing or not set. Please provide a valid API key.")
        st.error("If you need Context Explanation: use Gemini with an API key.")
        return None

    genai.configure(api_key=google_api_key)

    model = genai.GenerativeModel("gemini-2.0-flash-lite")

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None


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



st.set_page_config(page_title="🌍 Multilingual Translator with Context", page_icon="🌐", layout="centered")

st.title("🌍 Multilingual Translator with Context")
st.write("Translate text into different languages and get cultural/linguistic explanations.")


# if selected_provider == "Other (Custom)":
#     custom_llm_name = st.sidebar.text_input("🔤 Custom LLM Name")
#     custom_base_url = st.sidebar.text_input("🌐 Base URL")
#     model_options = ["Other"]
#     selected_model_option = "Other"
# else:
#     provider_info = LLM_MODELS[selected_provider]
#     model_options = provider_info["model"] + ["Other"]
#     selected_model_option = st.sidebar.selectbox("Select Model Version", model_options)

# If model is "Other" → ask for input
    # if selected_model_option == "Other":
    #     custom_model = st.sidebar.text_input("🧠 Enter Custom Model Name")
    # else:
    #     custom_model = selected_model_option

# Get API Key
# if selected_provider == "Other (Custom)" or LLM_MODELS.get(selected_provider, {}).get("requires_key", False):
#     user_api_key = st.sidebar.text_input("🔑 API Key", type="password")
# else:
#     env_key = LLM_MODELS[selected_provider]["api_key_env"]
#     user_api_key = os.getenv(env_key)
#
# # Get Base URL
# base_url = custom_base_url if selected_provider == "Other (Custom)" else LLM_MODELS[selected_provider]["base_url"]
# api_key = user_api_key


# === Input Query ===
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