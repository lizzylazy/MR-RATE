import requests
# -----------------------------------------------------------------------
# API Configuration
# -----------------------------------------------------------------------

API_URL = "http://vl90864.usz.ch:8001/v1/chat/completions"
MODEL_NAME = "gpt-oss-20b"

session = requests.Session()


def fetch_llm_response(messages, sampling_params):
    """
    Sends a request to a local OpenAI-compatible API.
    """
    if hasattr(sampling_params, "__dict__"):
        params_dict = sampling_params.__dict__
    else:
        params_dict = sampling_params

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