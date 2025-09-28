# translator.py
from transformers import MarianMTModel, MarianTokenizer
import os
from openai import OpenAI

def load_model(src_lang="en", tgt_lang="hi"):
    """
    Load MarianMT model for translation.
    Example: English (en) → Hindi (hi)
    """
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
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

def openai_translate_text(text, src_lang="en", tgt_lang="hi", model="gpt-4o-mini"):
    """
    Translate text using OpenAI's GPT model via API.
    """
    client = OpenAI()
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
