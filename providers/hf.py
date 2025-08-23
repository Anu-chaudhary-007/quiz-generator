import os
import json
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

HF_API_URL_TEMPLATE = "https://api-inference.huggingface.co/models/{model_id}"

class HFError(Exception):
    pass

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def generate(prompt: str, token: str, model_id: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
    if not token:
        raise HFError("Missing HUGGINGFACE_API_TOKEN.")

    url = HF_API_URL_TEMPLATE.format(model_id=model_id)
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False
        }
    }
    r = requests.post(url, headers=headers, json=payload, timeout=90)

    if r.status_code == 503:
        # Model cold-start on Inference API — raise to trigger retry
        raise HFError("Model loading (503). Retrying...")

    if not r.ok:
        raise HFError(f"HF API error {r.status_code}: {r.text[:500]}")

    data = r.json()
    # Responses can vary (list/dict). Try common shapes first.
    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"].strip()
    # Fallback
    try:
        return json.dumps(data)  # so we can see what came back in step 1
    except Exception:
        return str(data)

