#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mechanistic Probing Module: Teacher-Forced Sequence Scoring
Computes log-odds for Yes/No decisions to distinguish deep revision from superficial compliance.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LogOddsResult:
    """Result of teacher-forced log-odds computation"""
    lp_yes: float
    lp_no: float
    log_odds_yes_over_no: float
    decision: str  # "Yes" or "No"
    variants_aggregated: bool  # Whether multiple tokenization variants were aggregated


def sequence_logprob(model, tokenizer, prompt: str, completion: str, device: str = "cuda") -> float:
    """
    Teacher-forced log P(completion | prompt).
    
    Args:
        model: HuggingFace model with .generate() capability
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt text
        completion: Target completion text
        device: Device to run on ("cuda" or "cpu")
    
    Returns:
        float: Log probability of completion given prompt
    """
    full_text = prompt + completion
    full = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(device)
    pref = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    full_ids = full["input_ids"]  # [1, L]
    pref_len = pref["input_ids"].shape[1]  # prompt length
    target_len = full_ids.shape[1] - pref_len
    
    if target_len <= 0:
        raise ValueError(f"Completion tokenizes to {target_len} tokens, must be > 0")

    with torch.no_grad():
        logits = model(input_ids=full_ids).logits  # [1, L, V]

    # logits at position i predicts token i+1
    start = pref_len - 1
    end = full_ids.shape[1] - 1
    target_logits = logits[:, start:end, :]  # [1, target_len, V]
    target_ids = full_ids[:, pref_len:pref_len+target_len]  # [1, target_len]

    log_probs = F.log_softmax(target_logits, dim=-1)
    token_logps = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
    
    return token_logps.sum().item()


def aggregate_tokenization_variants(
    model, 
    tokenizer, 
    prompt: str, 
    base_text: str,
    variants: Optional[List[str]] = None,
    device: str = "cuda"
) -> Tuple[float, bool]:
    """
    Aggregate log probabilities over multiple tokenization variants.
    
    This handles cases where "Yes" might tokenize as " Yes", "Yes", "Yes\n", etc.
    We compute log P for each variant and aggregate using logsumexp.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt
        base_text: Base text (e.g., "Yes" or "No")
        variants: List of variants to try. If None, uses default set.
        device: Device to run on
    
    Returns:
        Tuple of (aggregated_log_prob, whether_aggregated)
    """
    if variants is None:
        # Default variants for Yes/No
        variants = [
            base_text,              # "Yes" or "No"
            f" {base_text}",        # " Yes" or " No"
            f"{base_text}\n",       # "Yes\n" or "No\n"
            f" {base_text}\n",      # " Yes\n" or " No\n"
            f"{base_text}.",        # "Yes." or "No."
            f"{base_text}.\n",      # "Yes.\n" or "No.\n"
        ]
    
    log_probs = []
    successful_variants = []
    
    for variant in variants:
        try:
            lp = sequence_logprob(model, tokenizer, prompt, variant, device)
            log_probs.append(lp)
            successful_variants.append(variant)
        except (ValueError, RuntimeError):
            # Variant failed (e.g., empty tokenization), skip
            continue
    
    if not log_probs:
        raise ValueError(f"All tokenization variants failed for base_text='{base_text}'")
    
    # Aggregate using logsumexp
    log_probs_tensor = torch.tensor(log_probs)
    aggregated_lp = torch.logsumexp(log_probs_tensor, dim=0).item()
    
    aggregated = len(successful_variants) > 1
    
    return aggregated_lp, aggregated


def yes_no_logodds(
    model, 
    tokenizer, 
    prompt: str,
    aggregate_variants: bool = True,
    device: str = "cuda"
) -> LogOddsResult:
    """
    Compute log-odds of Yes vs No given a prompt.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt text
        aggregate_variants: Whether to aggregate over tokenization variants
        device: Device to run on
    
    Returns:
        LogOddsResult with log probabilities and decision
    """
    if aggregate_variants:
        lp_yes, yes_agg = aggregate_tokenization_variants(model, tokenizer, prompt, "Yes", device=device)
        lp_no, no_agg = aggregate_tokenization_variants(model, tokenizer, prompt, "No", device=device)
        variants_aggregated = yes_agg or no_agg
    else:
        # Simple single-variant scoring
        lp_yes = sequence_logprob(model, tokenizer, prompt, "Yes\n", device)
        lp_no = sequence_logprob(model, tokenizer, prompt, "No\n", device)
        variants_aggregated = False
    
    log_odds = lp_yes - lp_no
    decision = "Yes" if lp_yes > lp_no else "No"
    
    return LogOddsResult(
        lp_yes=lp_yes,
        lp_no=lp_no,
        log_odds_yes_over_no=log_odds,
        decision=decision,
        variants_aggregated=variants_aggregated
    )


@dataclass
class RevisionTaxonomy:
    """Classification of revision type based on textual and mechanistic changes"""
    textual_reversal: bool
    sign_flip: bool
    latent_shift_magnitude: float
    latent_shift_exceeds_threshold: bool
    
    @property
    def category(self) -> str:
        """Classify into Deep / Superficial / Latent / Stable"""
        if self.textual_reversal and self.sign_flip:
            return "Deep"
        elif self.textual_reversal and not self.sign_flip:
            return "Superficial"
        elif not self.textual_reversal and self.latent_shift_exceeds_threshold:
            return "Latent"
        else:
            return "Stable"


def classify_revision(
    pass1_answer: str,
    pass2_answer: str,
    pass1_logodds: float,
    pass2_logodds: float,
    latent_threshold: float
) -> RevisionTaxonomy:
    """
    Classify revision type based on textual and mechanistic signals.
    
    Args:
        pass1_answer: Textual answer from Pass 1 ("Yes" or "No")
        pass2_answer: Textual answer from Pass 2 ("Yes" or "No")
        pass1_logodds: Log-odds(Yes/No) from Pass 1
        pass2_logodds: Log-odds(Yes/No) from Pass 2
        latent_threshold: Threshold for latent revision (e.g., 95th percentile of drift)
    
    Returns:
        RevisionTaxonomy object
    """
    textual_reversal = (pass1_answer != pass2_answer)
    
    # Sign flip: LogOdds changes sign
    sign_flip = (pass1_logodds * pass2_logodds < 0)
    
    # Latent shift
    delta_logodds = abs(pass2_logodds - pass1_logodds)
    latent_exceeds = (delta_logodds > latent_threshold)
    
    return RevisionTaxonomy(
        textual_reversal=textual_reversal,
        sign_flip=sign_flip,
        latent_shift_magnitude=delta_logodds,
        latent_shift_exceeds_threshold=latent_exceeds
    )


def compute_drift_calibrated_threshold(
    drift_trials: List[Dict],
    percentile: float = 95.0
) -> float:
    """
    Compute drift-calibrated latent threshold from A0 (drift baseline) trials.
    
    Args:
        drift_trials: List of A0 trials with 'delta_logodds' field
        percentile: Percentile to use (default: 95)
    
    Returns:
        float: Threshold value (95th percentile of |ΔLogOdds| under drift)
    """
    import numpy as np
    
    if not drift_trials:
        raise ValueError("No drift trials provided")
    
    delta_values = [abs(trial['delta_logodds']) for trial in drift_trials if 'delta_logodds' in trial]
    
    if not delta_values:
        raise ValueError("No delta_logodds values found in drift trials")
    
    threshold = np.percentile(delta_values, percentile)
    return float(threshold)


def load_model_for_probing(model_id: str, device: str = "cuda") -> Tuple:
    """
    Load a HuggingFace model and tokenizer for mechanistic probing.
    
    Args:
        model_id: Model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        device: Device to load on
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Global cache to prevent reloading
    global _MODEL_CACHE
    if not '_MODEL_CACHE' in globals():
        _MODEL_CACHE = {}
    
    if model_id in _MODEL_CACHE:
        # print(f"Using cached model: {model_id}")  # Optional: skip printing to reduce noise
        return _MODEL_CACHE[model_id]

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Check for bitsandbytes
    try:
        import bitsandbytes
        has_bnb = True
        print("bitsandbytes detected - enabling 8-bit quantization")
    except ImportError:
        has_bnb = False
        print("bitsandbytes not found - using default precision")
        
    model_kwargs = {
        "device_map": "auto" if device == "cuda" else None,
    }
    
    if has_bnb and device == "cuda":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        model_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_kwargs
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    
    print(f"Model loaded on {device}")
    _MODEL_CACHE[model_id] = (model, tokenizer)
    return model, tokenizer


# Example usage
if __name__ == "__main__":
    # This is a reference implementation demonstrating the API
    print("Mechanistic Probing Module")
    print("=" * 60)
    print("This module provides teacher-forced sequence scoring for Yes/No decisions.")
    print("Use yes_no_logodds() to compute log-odds from a prompt.")
    print("Use classify_revision() to categorize revision types.")
    print("=" * 60)

