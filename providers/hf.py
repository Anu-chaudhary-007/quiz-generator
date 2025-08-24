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
        
        # Check if the model is still loading
        if response.status_code == 503:
            # Extract estimated time if available
            est_time = "unknown"
            try:
                data = response.json()
                if "estimated_time" in data:
                    est_time = f"{data['estimated_time']:.1f}"
            except:
                pass
            raise HFError(f"Model is loading. Please try again in {est_time} seconds.")
        
        response.raise_for_status()
        data = response.json()

        # Handle Hugging Face error message
        if isinstance(data, dict) and "error" in data:
            raise HFError(data["error"])

        # Handle different response formats from different models
        if isinstance(data, list):
            # For text generation models like google/flan-t5-base
            if len(data) > 0 and "generated_text" in data[0]:
                return data[0]["generated_text"]
            # For other models that might return a list of results
            elif len(data) > 0 and isinstance(data[0], dict) and "label" in data[0]:
                return str(data[0])
        
        # For models that return a dictionary directly
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
            
        # Fallback - return the raw data as string
        return str(data)

    except requests.exceptions.RequestException as e:
        raise HFError(f"Request failed: {e}")
