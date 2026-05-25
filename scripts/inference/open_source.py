"""
Open-source model implementations.
"""
from scripts.inference import OpenSourceModelInference


class MistralInference(OpenSourceModelInference):
    """Mistral-7B-Instruct-v0.3 inference."""

    def __init__(self, device: str = "cuda", dtype: str = "float16", quantization: str = None):
        super().__init__(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            device=device,
            dtype=dtype,
            quantization=quantization,
        )
        self.model_name = "Mistral-7B-Instruct-v0.3"


class Qwen25Inference(OpenSourceModelInference):
    """Qwen2.5-7B-Instruct inference."""

    def __init__(self, device: str = "cuda", dtype: str = "float16", quantization: str = None):
        super().__init__(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            device=device,
            dtype=dtype,
            quantization=quantization,
        )
        self.model_name = "Qwen2.5-7B-Instruct"


class Qwen3Inference(OpenSourceModelInference):
    """Qwen3 inference (maps to Qwen3.5-9B checkpoint)."""

    def __init__(self, device: str = "cuda", dtype: str = "float16", quantization: str = None):
        super().__init__(
            model_id="Qwen/Qwen3.5-9B",
            device=device,
            dtype=dtype,
            quantization=quantization,
        )
        self.model_name = "Qwen3.5-9B"


class Llama31Inference(OpenSourceModelInference):
    """Llama-3.1-8B-Instruct inference."""

    def __init__(self, device: str = "cuda", dtype: str = "float16", quantization: str = None):
        super().__init__(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            device=device,
            dtype=dtype,
            quantization=quantization,
        )
        self.model_name = "Llama-3.1-8B-Instruct"


class Gemma29BInference(OpenSourceModelInference):
    """Gemma-2-9b-it inference."""

    def __init__(self, device: str = "cuda", dtype: str = "float16", quantization: str = None):
        super().__init__(
            model_id="google/gemma-2-9b-it",
            device=device,
            dtype=dtype,
            quantization=quantization,
        )
        self.model_name = "Gemma-2-9b-it"


def get_open_source_model(model_name: str, device: str = "cuda", 
                          dtype: str = "float16", quantization: str = None) -> OpenSourceModelInference:
    """Factory function to get an open-source model by name."""
    models = {
        "Mistral-7B-Instruct-v0.3": MistralInference,
        "Qwen2.5-7B-Instruct": Qwen25Inference,
        "Qwen3.5-9B": Qwen3Inference,
        "Llama-3.1-8B-Instruct": Llama31Inference,
        "Gemma-2-9b-it": Gemma29BInference,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](device=device, dtype=dtype, quantization=quantization)
