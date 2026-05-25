"""
Base classes and interfaces for model inference.
"""
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ModelInference(ABC):
    """Abstract base class for model inference."""

    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "float16"):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.loaded = False

    @abstractmethod
    def load(self):
        """Load the model."""
        pass

    @abstractmethod
    def unload(self):
        """Unload the model to free memory."""
        pass

    @abstractmethod
    def infer(self, prompt: str, temperature: float = 0.0, max_tokens: int = 50) -> str:
        """Run inference on a prompt."""
        pass

    def infer_batch(self, prompts: List[str], temperature: float = 0.0, max_tokens: int = 50) -> List[str]:
        """Run inference on a batch of prompts."""
        return [self.infer(prompt, temperature=temperature, max_tokens=max_tokens) for prompt in prompts]

    @abstractmethod
    def is_refusal(self, text: str) -> bool:
        """Detect if the model refused to answer."""
        pass

    def format_tensor_dtype(self):
        """Convert string dtype to torch dtype."""
        import torch
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.dtype, torch.float16)


class OpenSourceModelInference(ModelInference):
    """Base class for open-source model inference."""

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "float16", 
                 quantization: Optional[str] = None):
        super().__init__(model_id, device, dtype)
        self.model_id = model_id
        self.quantization = quantization
        self.tokenizer = None
        self.model = None

    def _requires_remote_code(self) -> bool:
        """Return True for model families that need trust_remote_code."""
        return self.model_id.startswith("Qwen/Qwen3")

    def load(self):
        """Load tokenizer and model."""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info(f"Loading {self.model_name}...")

        tokenizer_kwargs = {}
        if self._requires_remote_code():
            tokenizer_kwargs["trust_remote_code"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **tokenizer_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Configure loading based on quantization
        device_map = self.device
        if isinstance(self.device, str) and self.device.startswith("cuda:"):
            # Keep the full model on one explicit GPU when a concrete CUDA device is given.
            device_map = {"": self.device}

        kwargs = {
            "device_map": device_map,
            "low_cpu_mem_usage": True,
            "torch_dtype": self.format_tensor_dtype(),
        }

        if self._requires_remote_code():
            kwargs["trust_remote_code"] = True

        if self.quantization == "4bit":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.format_tensor_dtype(),
            )
            kwargs["quantization_config"] = bnb_config
        elif self.quantization == "8bit":
            kwargs["load_in_8bit"] = True

        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        except RuntimeError as exc:
            error_message = str(exc).lower()
            can_retry_8bit = self.quantization is None and "out of memory" in error_message
            if not can_retry_8bit:
                raise

            logger.warning(
                "CUDA OOM while loading %s in %s. Retrying with 8-bit quantization.",
                self.model_name,
                self.dtype,
            )
            from transformers import BitsAndBytesConfig

            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("torch_dtype", None)
            retry_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **retry_kwargs)

        self.model.eval()
        self.loaded = True
        logger.info(f"Loaded {self.model_name}")

    def unload(self):
        """Unload model."""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        self.loaded = False
        logger.info(f"Unloaded {self.model_name}")

    def infer(self, prompt: str, temperature: float = 0.0, max_tokens: int = 50) -> str:
        """Run inference."""
        if not self.loaded:
            self.load()

        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def infer_batch(self, prompts: List[str], temperature: float = 0.0, max_tokens: int = 50) -> List[str]:
        """Run batched inference for open-source models."""
        if not self.loaded:
            self.load()

        messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]
        formatted_prompts = [
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_list
        ]

        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        prompt_length = inputs["input_ids"].shape[1]

        import torch

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = []
        for batch_index in range(len(prompts)):
            generated = outputs[batch_index][prompt_length:]
            decoded.append(self.tokenizer.decode(generated, skip_special_tokens=True).strip())

        return decoded

    def is_refusal(self, text: str) -> bool:
        """Detect refusal patterns."""
        refusal_patterns = [
            "i don't know",
            "i cannot",
            "i can't",
            "i'm unable",
            "i don't have",
            "i don't possess",
            "unclear",
            "don't understand",
            "i'm sorry",
            "apologize",
        ]
        text_lower = text.lower().strip()
        return any(pattern in text_lower for pattern in refusal_patterns)
