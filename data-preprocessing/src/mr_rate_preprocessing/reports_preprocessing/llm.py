# =======================================================================
# Client for a local OpenAI-compatible LLM server.
#
# Lets the reports preprocessing steps talk to a model served by e.g.
#   vllm serve /data/dl_models/Qwen/Qwen3.5-35B-A3B-FP8 \
#       --served-model-name Qwen3.5-35B-A3B-FP8 --port 8000
# instead of loading vLLM in-process. Importing this module does not
# require vLLM, so driver scripts can run on a CPU-only login node.
# =======================================================================

import os
import threading
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_API_URL = os.environ.get(
    "MR_RATE_LLM_API_URL", "http://localhost:8000/v1/chat/completions"
)
DEFAULT_MODEL_NAME = os.environ.get("MR_RATE_LLM_MODEL", "Qwen3.5-35B-A3B-FP8")
DEFAULT_TIMEOUT = float(os.environ.get("MR_RATE_LLM_TIMEOUT", "300"))


@dataclass
class SamplingParams:
    """Subset of vllm.SamplingParams the chat completions endpoint accepts.

    Provided so callers do not have to import vLLM just to specify decoding
    parameters; vllm.SamplingParams is also accepted by fetch_llm_response.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048


class LLMError(RuntimeError):
    """Raised when the server cannot be reached or returns an unusable reply."""


# requests.Session is not thread-safe, and the callers use thread pools, so
# each thread gets its own session and connection pool.
_thread_local = threading.local()


def _session():
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        # Retry transient server-side and connection errors. 400 (context
        # length exceeded, bad request) is not retried -- it never succeeds.
        retry = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"POST"}),
        )
        session.mount("http://", HTTPAdapter(max_retries=retry))
        session.mount("https://", HTTPAdapter(max_retries=retry))
        _thread_local.session = session
    return session


def fetch_llm_response(
    messages,
    sampling_params,
    api_url=None,
    model_name=None,
    timeout=DEFAULT_TIMEOUT,
):
    """Send one chat completion request and return the reply text.

    Raises LLMError instead of returning a sentinel string: a failed call
    must not be written to disk as if it were a model output.
    """
    payload = {
        "model": model_name or DEFAULT_MODEL_NAME,
        "messages": messages,
        "temperature": getattr(sampling_params, "temperature", 0.0),
        "top_p": getattr(sampling_params, "top_p", 1.0),
        "max_tokens": getattr(sampling_params, "max_tokens", 2048),
    }

    url = api_url or DEFAULT_API_URL

    try:
        response = _session().post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
            # The server is local; never route it through an HTTP(S) proxy.
            proxies={"http": None, "https": None},
        )
    except requests.RequestException as exc:
        raise LLMError(f"request to {url} failed: {exc}") from exc

    if response.status_code >= 400:
        # The body carries the actual reason (context length, unknown model).
        raise LLMError(f"{url} returned HTTP {response.status_code}: {response.text[:500]}")

    try:
        message = response.json()["choices"][0]["message"]
    except (ValueError, KeyError, IndexError) as exc:
        raise LLMError(f"unexpected response from {url}: {response.text[:500]}") from exc

    # Reasoning models put the answer in `content` and the chain of thought in
    # `reasoning`; fall back to the latter only when `content` is empty.
    text = message.get("content") or message.get("reasoning") or ""
    return text.strip()
