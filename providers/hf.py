import requests

class HFError(Exception):
    """Custom error class for Hugging Face API failures."""
    pass


def generate(prompt: str, hf_token: str, model_id: str) -> str:
    """
    Calls the Hugging Face Inference API with the given prompt.

    Args:
        prompt (str): The input text / instruction.
        hf_token (str): Hugging Face API token (hf_xxx).
        model_id (str): The Hugging Face model repo ID.

    Returns:
        str: The generated text from the model.

    Raises:
        HFError: If the API call fails or Hugging Face returns an error.
    """
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    try:
        response = requests.post(
            url,
            headers=headers,
            json={"inputs": prompt},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        # Handle Hugging Face error message
        if isinstance(data, dict) and "error" in data:
            raise HFError(data["error"])

        # Hugging Face text generation models usually return a list
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]

        # Fallback
        return str(data)

    except requests.exceptions.RequestException as e:
        raise HFError(f"Request failed: {e}")
