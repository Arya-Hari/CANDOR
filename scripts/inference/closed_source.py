"""
Closed-source model implementations (API-based).
Cleaned and consolidated: OpenAI GPT-5.1, Google Gemini, DeepSeek (Bedrock),
and Meta Llama 3.3 (Bedrock Converse/profile aware).
"""
import os
import json
import logging
import random
import time
from urllib.parse import urlsplit, parse_qs
from typing import List, Optional

import requests
from scripts.inference import ModelInference

logger = logging.getLogger(__name__)


REFUSAL_PATTERNS = [
    "i don't know",
    "i cannot",
    "i can't",
    "i'm unable",
    "i don't have",
    "i'm not able",
    "i do not",
    "i'm sorry",
    "apologize",
]


def is_refusal_text(text: str) -> bool:
    """Return True when the text looks like a refusal."""
    normalized = (text or "").lower().strip()
    return any(pattern in normalized for pattern in REFUSAL_PATTERNS)


def first_env_value(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def parse_azure_openai_endpoint(endpoint: str) -> tuple[str, Optional[str]]:
    """Normalize Azure OpenAI endpoint inputs and extract api-version if embedded."""
    parsed = urlsplit(endpoint)
    azure_endpoint = f"{parsed.scheme}://{parsed.netloc}"
    query = parse_qs(parsed.query)
    api_version = None
    if query.get("api-version"):
        api_version = query["api-version"][0]
    return azure_endpoint, api_version


class GPT5_1Inference(ModelInference):
    """GPT-5.1 inference via OpenAI API."""

    def __init__(self):
        super().__init__("GPT-5.1", device="api", dtype="N/A")
        self.client = None
        self.backend = None
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.azure_api_key = first_env_value("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_APIP_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    def load(self):
        if self.azure_api_key and self.azure_endpoint:
            from openai import AzureOpenAI

            endpoint, api_version_from_endpoint = parse_azure_openai_endpoint(self.azure_endpoint)
            api_version = self.azure_api_version or api_version_from_endpoint or "2025-04-01-preview"
            self.client = AzureOpenAI(api_key=self.azure_api_key, azure_endpoint=endpoint, api_version=api_version)
            self.backend = "azure"
            self.loaded = True
            logger.info("Loaded GPT-5.1 via Azure OpenAI at %s (api_version=%s)", endpoint, api_version)
            return

        if self.api_key:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)
            self.backend = "openai"
            self.loaded = True
            logger.info("Loaded GPT-5.1 via OpenAI")
            return

        raise ValueError("Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY/AZURE_OPENAI_ENDPOINT in environment")

    def unload(self):
        self.client = None
        self.backend = None
        self.loaded = False

    def is_refusal(self, text: str) -> bool:
        return is_refusal_text(text)

    def infer(self, prompt: str, temperature: float = 0.0, max_tokens: int = 50) -> str:
        if not self.loaded:
            self.load()

        completion_tokens = max(max_tokens, 16)
        if completion_tokens != max_tokens:
            logger.warning(
                "GPT-5.1 requested max_tokens=%s; using %s to avoid finish failures.",
                max_tokens,
                completion_tokens,
            )

        if self.backend == "azure":
            response = self.client.responses.create(
                model="gpt-5.1",
                input=prompt,
                max_output_tokens=completion_tokens,
                temperature=temperature,
            )
            text = getattr(response, "output_text", None)
            if text:
                return text.strip()
            return str(response).strip()

        response = self.client.chat.completions.create(
            model="gpt-5.1",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=completion_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()


class GPT4_1Inference(ModelInference):
    """GPT-4.1 inference via OpenAI API."""

    def __init__(self):
        super().__init__("GPT-4.1", device="api", dtype="N/A")
        self.client = None
        self.backend = None
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.azure_api_key = first_env_value("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_APIP_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    def load(self):
        if self.azure_api_key and self.azure_endpoint:
            from openai import AzureOpenAI

            endpoint, api_version_from_endpoint = parse_azure_openai_endpoint(self.azure_endpoint)
            api_version = self.azure_api_version or api_version_from_endpoint or "2025-04-01-preview"
            self.client = AzureOpenAI(api_key=self.azure_api_key, azure_endpoint=endpoint, api_version=api_version)
            self.backend = "azure"
            self.loaded = True
            logger.info("Loaded GPT-4.1 via Azure OpenAI at %s (api_version=%s)", endpoint, api_version)
            return

        if self.api_key:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)
            self.backend = "openai"
            self.loaded = True
            logger.info("Loaded GPT-4.1 via OpenAI")
            return

        raise ValueError("Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY/AZURE_OPENAI_ENDPOINT in environment")

    def unload(self):
        self.client = None
        self.backend = None
        self.loaded = False

    def is_refusal(self, text: str) -> bool:
        return is_refusal_text(text)

    def infer(self, prompt: str, temperature: float = 0.0, max_tokens: int = 50) -> str:
        if not self.loaded:
            self.load()

        completion_tokens = max(max_tokens, 16)
        if completion_tokens != max_tokens:
            logger.warning(
                "GPT-4.1 requested max_tokens=%s; using %s to avoid finish failures.",
                max_tokens,
                completion_tokens,
            )

        if self.backend == "azure":
            response = self.client.responses.create(
                model="gpt-4.1",
                input=prompt,
                max_output_tokens=completion_tokens,
                temperature=temperature,
            )
            text = getattr(response, "output_text", None)
            if text:
                return text.strip()
            return str(response).strip()

        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=completion_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()


class GeminiInference(ModelInference):
    """Gemini inference via Google API."""

    def __init__(self):
        super().__init__("Gemini-2.0-flash", device="api", dtype="N/A")
        self.model = None
        self.api_key = os.getenv("GEMINI_API_KEY")

    def load(self):
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-lite")
        self.loaded = True
        logger.info("Loaded Gemini")

    def unload(self):
        self.loaded = False

    def is_refusal(self, text: str) -> bool:
        return is_refusal_text(text)

    def infer(self, prompt: str, temperature: float = 0.0, max_tokens: int = 50) -> str:
        if not self.loaded:
            self.load()

        import google.generativeai as genai

        config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        response = self.model.generate_content(prompt, generation_config=config)
        return response.text.strip()


class Gemini25ProInference(ModelInference):
    """Gemini 2.5 Pro inference via Google Cloud Vertex AI using ADC."""

    def __init__(self):
        super().__init__("Gemini-2.5-pro", device="api", dtype="N/A")
        self.client = None
        self.project_id = (
            os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("GOOGLE_PROJECT_ID")
            or os.getenv("PROJECT_ID")
            or os.getenv("GCLOUD_PROJECT")
        )
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("VERTEX_AI_LOCATION") or "us-central1"

    def load(self):
        if not self.project_id:
            try:
                import google.auth

                _, project_id = google.auth.default()
                self.project_id = project_id
            except Exception:
                self.project_id = None

        if not self.project_id:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT (or GOOGLE_PROJECT_ID/PROJECT_ID/GCLOUD_PROJECT) not set in environment"
            )

        try:
            from google import genai
        except ImportError as exc:
            raise ImportError("google-genai is required for Gemini-2.5-pro support") from exc

        self.client = genai.Client(vertexai=True, project=self.project_id, location=self.location)
        self.loaded = True
        logger.info("Loaded Gemini-2.5-pro via Vertex AI in project %s (%s)", self.project_id, self.location)

    def unload(self):
        self.client = None
        self.loaded = False

    def is_refusal(self, text: str) -> bool:
        return is_refusal_text(text)

    def infer(self, prompt: str, temperature: float = 0.0, max_tokens: int = 50) -> str:
        if not self.loaded:
            self.load()

        from google.genai import types

        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        response = self.client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=config,
        )

        text = getattr(response, "text", None)
        if text:
            return text.strip()

        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if not parts:
                continue
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    return part_text.strip()

        return str(response).strip()


class DeepSeekInference(ModelInference):
    """Legacy DeepSeek API wrapper (kept for compatibility)."""

    def __init__(self):
        super().__init__("DeepSeek-v3.2", device="api", dtype="N/A")
        self.client = None
        self.api_key = os.getenv("DEEPSEEK_API_KEY")

    def load(self):
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not set in environment")
        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        self.loaded = True
        logger.info("Loaded DeepSeek (Open API)")

    def unload(self):
        self.loaded = False

    def is_refusal(self, text: str) -> bool:
        return is_refusal_text(text)

    def infer(self, prompt: str, temperature: float = 0.0, max_tokens: int = 50) -> str:
        if not self.loaded:
            self.load()

        response = self.client.chat.completions.create(
            model="DeepSeek-v3.2",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()


class BedrockModelInference(ModelInference):
    """Generic Bedrock model wrapper supporting both invoke_model and converse.

    - Prefers signed boto3 client (bedrock-runtime) when AWS creds exist.
    - Falls back to REST bearer token if `AWS_BEARER_TOKEN_BEDROCK` is set.
    - Region picked from BEDROCK_REGION -> AWS_REGION -> default ('us-east-1').
    """

    def __init__(self, model_name: str, model_ids: List[str], region: Optional[str] = None):
        super().__init__(model_name, device="api", dtype="N/A")
        self.region = region or os.getenv("BEDROCK_REGION") or os.getenv("AWS_REGION") or "us-east-1"
        self.model_ids = model_ids
        self.bearer = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        self.boto3_client = None

    def load(self):
        try:
            import boto3
            aws_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
            # bedrock-runtime is used for invocation; bedrock is used for profile/listing
            if aws_key and aws_secret:
                self.boto3_client = __import__("boto3").client("bedrock-runtime", region_name=self.region)
        except Exception:
            self.boto3_client = None

        if not self.boto3_client and not self.bearer:
            raise ValueError("No AWS credentials or AWS_BEARER_TOKEN_BEDROCK found in environment")

        self.loaded = True
        logger.info("Loaded Bedrock model '%s' in region %s", self.model_name, self.region)

    def unload(self):
        """Unload Bedrock client state."""
        try:
            if self.boto3_client:
                # boto3 clients don't need explicit close, but remove reference
                del self.boto3_client
        except Exception:
            pass
        self.boto3_client = None
        self.loaded = False
        logger.info("Unloaded Bedrock model '%s'", self.model_name)

    def is_refusal(self, text: str) -> bool:
        """Simple refusal detector re-using common patterns."""
        refusal_patterns = [
            "i don't know",
            "i cannot",
            "i can't",
            "i'm unable",
            "i don't have",
            "i'm not able",
            "i do not",
            "i'm sorry",
            "apologize",
        ]
        txt = (text or "").lower().strip()
        return any(p in txt for p in refusal_patterns)

    def _is_retryable_bedrock_error(self, exc: Exception) -> bool:
        message = str(exc)
        retryable_markers = (
            "ThrottlingException",
            "TooManyRequestsException",
            "Too many requests",
            "Rate exceeded",
            "ProvisionedThroughputExceededException",
        )
        return any(marker in message for marker in retryable_markers)

    def _retry_bedrock_call(self, operation_name: str, model_id: str, call_fn, max_attempts: int = 5):
        last_err = None
        for attempt in range(max_attempts):
            try:
                return call_fn()
            except Exception as exc:
                last_err = exc
                if attempt >= max_attempts - 1 or not self._is_retryable_bedrock_error(exc):
                    raise
                delay = min(16.0, (2 ** attempt) + random.uniform(0.0, 1.0))
                logger.warning(
                    "Bedrock %s throttled for %s on %s; retrying in %.1fs (%s/%s)",
                    operation_name,
                    model_id,
                    self.model_name,
                    delay,
                    attempt + 1,
                    max_attempts,
                )
                time.sleep(delay)
        raise last_err

    def _invoke_via_boto3(self, model_id: str, payload: dict) -> str:
        body = json.dumps(payload).encode("utf-8")
        resp = self.boto3_client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        stream = resp.get("body")
        if hasattr(stream, "read"):
            return stream.read().decode("utf-8")
        if isinstance(stream, (bytes, bytearray)):
            return stream.decode("utf-8")
        return str(stream)

    def _invoke_via_rest(self, model_id: str, payload: dict) -> str:
        endpoint = f"https://bedrock-runtime.{self.region}.amazonaws.com/models/{model_id}/invoke"
        headers = {
            "Authorization": f"Bearer {self.bearer}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        response = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=30)
        response.raise_for_status()
        return response.text

    def infer(self, prompt: str, temperature: float = 0.0, max_tokens: int = 50) -> str:
        """Default invoke_model chat-style path (used by DeepSeek-like models)."""
        if not self.loaded:
            self.load()

        payload = {"messages": [{"role": "user", "content": prompt}], "parameters": {"max_output_tokens": max_tokens, "temperature": temperature}}

        last_err = None
        for model_id in self.model_ids:
            try:
                logger.info(
                    "Bedrock request started: model=%s model_id=%s backend=%s prompt_chars=%s max_tokens=%s temperature=%.2f",
                    self.model_name,
                    model_id,
                    "boto3" if self.boto3_client else "rest",
                    len(prompt),
                    max_tokens,
                    temperature,
                )
                if self.boto3_client:
                    out = self._retry_bedrock_call("invoke_model", model_id, lambda: self._invoke_via_boto3(model_id, payload))
                else:
                    out = self._invoke_via_rest(model_id, payload)
                try:
                    parsed = json.loads(out)
                    if isinstance(parsed, dict):
                        # common chat response shapes
                        if "choices" in parsed and parsed["choices"]:
                            c = parsed["choices"][0]
                            if isinstance(c, dict) and "message" in c:
                                return c["message"].get("content", "").strip()
                        for key in ("output", "body", "content", "result", "generated_text", "text"):
                            if key in parsed and parsed[key] is not None:
                                text = str(parsed[key]).strip()
                                logger.info(
                                    "Bedrock request finished: model=%s model_id=%s response_chars=%s",
                                    self.model_name,
                                    model_id,
                                    len(text),
                                )
                                return text
                except Exception:
                    text = out.strip()
                    logger.info(
                        "Bedrock request finished: model=%s model_id=%s response_chars=%s",
                        self.model_name,
                        model_id,
                        len(text),
                    )
                    return text
            except Exception as e:
                last_err = e
                logger.debug("Bedrock model %s failed: %s", model_id, e)

        raise RuntimeError(f"All Bedrock model invocations failed for {self.model_name}. Last error: {last_err}")


class BedrockDeepSeekInference(BedrockModelInference):
    """Adapter for DeepSeek-style chat models on Bedrock."""

    def __init__(self, region: Optional[str] = None, model_ids: Optional[List[str]] = None):
        candidates = model_ids or ["deepseek.v3.2", "deepseek-v3-2", "deepseek-v3.2"]
        super().__init__(model_name="DeepSeek-Bedrock", model_ids=candidates, region=region)


class BedrockLlama33_70BInference(BedrockModelInference):
    """Meta Llama 3.3 70B Instruct via Bedrock Converse/profile.

    Uses the Converse-style invocation (boto3 `converse`) against the
    inference profile/ARn when available. The `model_ids` should contain the
    profile id (for example `us.meta.llama3-3-70b-instruct-v1:0`).
    """

    def __init__(self, region: Optional[str] = None, profile_ids: Optional[List[str]] = None):
        # prefer explicit profile ids when provided; the on-demand base model ID is not valid here
        candidates = profile_ids or ["us.meta.llama3-3-70b-instruct-v1:0"]
        super().__init__(model_name="meta.llama3-3-70b-instruct-v1:0", model_ids=candidates, region=region)

    def infer(self, prompt: str, temperature: float = 0.0, max_tokens: int = 50) -> str:
        if not self.loaded:
            self.load()

        # Converse-style requires boto3 signed client (we call converse on bedrock-runtime)
        if not self.boto3_client:
            raise RuntimeError("Bedrock Llama inference requires boto3/bedrock-runtime client with AWS credentials")

        last_err = None
        for mid in self.model_ids:
            try:
                # `converse` signature varies; this uses the typical structure returned by boto3
                resp = self._retry_bedrock_call(
                    "converse",
                    mid,
                    lambda: self.boto3_client.converse(
                        modelId=mid,
                        messages=[{"role": "user", "content": [{"text": prompt}]}],
                        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
                    ),
                )
                out = resp.get("output") or resp
                if isinstance(out, dict):
                    messages_out = out.get("messages") or []
                    if messages_out and isinstance(messages_out, list):
                        for m in messages_out:
                            content = m.get("content") or []
                            if isinstance(content, list) and content:
                                first = content[0]
                                if isinstance(first, dict) and first.get("text"):
                                    return first.get("text").strip()

                    msg = out.get("message") or {}
                    content = msg.get("content") or []
                    if content and isinstance(content, list):
                        first = content[0]
                        if isinstance(first, dict) and first.get("text"):
                            return first.get("text").strip()
                    for k in ("text", "generated_text", "content"):
                        if k in out and out[k]:
                            return str(out[k]).strip()
                return str(resp).strip()
            except Exception as exc:
                last_err = exc
                logger.debug("Bedrock Llama converse failed for %s: %s", mid, exc)

        raise RuntimeError(f"All Bedrock Llama invocations failed for {self.model_name}. Last error: {last_err}")


def get_closed_source_model(model_name: str) -> ModelInference:
    """Factory function to get a closed-source model by name."""
    models = {
        "GPT-5.1": GPT5_1Inference,
        "Gemini-2.0-flash": GeminiInference,
        "DeepSeek-v3.2": BedrockDeepSeekInference,
        "DeepSeek-Bedrock": BedrockDeepSeekInference,
        "Llama-3.3-70B": BedrockLlama33_70BInference,
        "meta.llama3-3-70b-instruct-v1:0": BedrockLlama33_70BInference,
        "us.meta.llama3-3-70b-instruct-v1:0": BedrockLlama33_70BInference,
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    # instantiate
    return models[model_name]()
