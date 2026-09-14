import requests
# -----------------------------------------------------------------------
# API Configuration
# -----------------------------------------------------------------------

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3.5-35B-A3B-FP8"

session = requests.Session()


def fetch_llm_response(messages, sampling_params):
    """
    Sends a request to a local OpenAI-compatible API.
    """
    params_dict = {
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "max_tokens": sampling_params.max_tokens,
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        **params_dict,
    }

    try:
        response = session.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300,
            proxies={"http": None, "https": None},
        )

        response.raise_for_status()
        data = response.json()

        message = data["choices"][0]["message"]

        text = (
                message.get("content")
                or message.get("reasoning")
                or ""
        )

        return text.strip()

    except Exception as e:
        print(f"[Error] API Call failed: {e}")
        return "[ERROR] API call failed"