import google.generativeai as genai

class GeminiError(Exception):
    pass

def generate(prompt: str, api_key: str, model_id: str = "gemini-1.5-flash", max_output_tokens: int = 512, temperature: float = 0.7) -> str:
    if not api_key:
        raise GeminiError("Missing GOOGLE_API_KEY.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_id)
    resp = model.generate_content(prompt, generation_config={"temperature": temperature, "max_output_tokens": max_output_tokens})
    if not resp or not resp.text:
        raise GeminiError("Empty response from Gemini.")
    return resp.text.strip()

