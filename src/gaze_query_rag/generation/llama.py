from __future__ import annotations

from pathlib import Path

import numpy as np

from gaze_query_rag.modeling.devices import resolve_torch_device
from gaze_query_rag.schemas import GeneratorBundle


def _resolve_dtype(dtype: str):
    import torch

    if dtype == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def load_llama_generator(
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    device: str = "auto",
    cache_dir: str | Path | None = None,
    dtype: str = "auto",
) -> GeneratorBundle:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = resolve_torch_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir) if cache_dir else None)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(cache_dir) if cache_dir else None,
        torch_dtype=_resolve_dtype(dtype),
    )
    model.to(resolved_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return GeneratorBundle(
        tokenizer=tokenizer,
        model=model,
        device=str(resolved_device),
        model_name=model_name,
    )


def generate_answer(
    bundle: GeneratorBundle,
    prompt: str,
    max_new_tokens: int = 8,
    temperature: float = 0.0,
) -> str:
    import torch

    encoded = bundle.tokenizer(prompt, return_tensors="pt").to(bundle.device)
    do_sample = temperature > 0
    with torch.inference_mode():
        output_ids = bundle.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=bundle.tokenizer.eos_token_id,
        )
    generated = output_ids[0, encoded["input_ids"].shape[1] :]
    return bundle.tokenizer.decode(generated, skip_special_tokens=True).strip()


def score_answer_options(
    bundle: GeneratorBundle,
    prompt_prefix: str,
    choices: list[str],
) -> np.ndarray:
    import torch

    if not choices:
        raise ValueError("choices must not be empty.")
    if bundle.tokenizer.pad_token_id is None:
        bundle.tokenizer.pad_token = bundle.tokenizer.eos_token
    full_texts = [prompt_prefix + choice for choice in choices]
    encoded_full = bundle.tokenizer(full_texts, return_tensors="pt", padding=True).to(bundle.device)
    encoded_prefix = bundle.tokenizer(prompt_prefix, return_tensors="pt").to(bundle.device)
    prefix_len = encoded_prefix["input_ids"].shape[1]
    input_ids = encoded_full["input_ids"]
    attention_mask = encoded_full["attention_mask"]
    with torch.inference_mode():
        logits = bundle.model(**encoded_full).logits
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    valid_targets = attention_mask[:, 1:].to(token_log_probs.dtype)
    start = max(prefix_len - 1, 0)
    valid_targets[:, :start] = 0
    scores = (token_log_probs * valid_targets).sum(dim=1)
    return scores.float().detach().cpu().numpy().astype(np.float64)
