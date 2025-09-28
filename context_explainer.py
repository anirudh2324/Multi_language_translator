# context_explainer.py
from transformers import MarianMTModel, MarianTokenizer
from openai import OpenAI

# Initialize OpenAI client (make sure OPENAI_API_KEY is set in your env)
client = OpenAI()


def load_model(src_lang="en", tgt_lang="hi"):
    """
    Load MarianMT model for translation.
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
        model="gpt-4o-mini",  # can use gpt-4, gpt-3.5, or a local LLM
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )

    return response.choices[0].message.content


# 🚀 Example usage
if __name__ == "__main__":
    text = "Break a leg!"
    translated = translate_text(text, src_lang="en", tgt_lang="hi")
    print(f"Original: {text}")
    print(f"Translated: {translated}")

    explanation = explain_context(text, translated, "en", "hi")
    print("\nContext Explanation:")
    print(explanation)
