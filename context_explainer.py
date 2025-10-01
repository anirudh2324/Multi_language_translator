from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import streamlit as st
from openai import OpenAI

OPENAI_API_KEY = st.secrets["api_keys"]["openai"]
client = OpenAI(api_key=OPENAI_API_KEY)


print("Loading M2M100 model (this may take a minute)...")
MODEL_NAME = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)
print("Model loaded successfully!")


def translate_text_multilingual(text, src_lang="en", tgt_lang="hi"):
    """
    Translate text from source language to target language using the preloaded M2M100 model.
    """
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang)
    )
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text


def explain_context(original_text, translated_text, src_lang="en", tgt_lang="hi"):
    """
    Use LLM to provide cultural/linguistic context for the translation.
    """
    prompt = f"""
Original text ({src_lang}): {original_text}
Translated text ({tgt_lang}): {translated_text}

Please explain:
1. Why this translation was chosen.
2. If the original text contains idioms, metaphors, or slang, clarify their cultural meaning.
3. Suggest alternative phrasings if available.
"""

    response = client.chat.completions.create(
        model="gpt-4",  # or gpt-3.5-turbo
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message.content



if __name__ == "__main__":
    text = "Hello, how are you?"
    src_lang = "en"
    tgt_lang = "hi"

    # Translate text
    translated = translate_text_multilingual(text, src_lang=src_lang, tgt_lang=tgt_lang)
    print(f"Translated Text ({tgt_lang}): {translated}")

    # Generate explanation
    explanation = explain_context(text, translated, src_lang=src_lang, tgt_lang=tgt_lang)
    print("\nCultural/Linguistic Explanation:\n", explanation)
