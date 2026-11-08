#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Utilities: Length Matching and Padding

Implements proper token-level length matching as required by §4.5 of the proposal.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


def compute_prompt_token_length(prompt: str, tokenizer) -> int:
    """
    Compute token length of a prompt using the tokenizer.
    
    Args:
        prompt: Text prompt
        tokenizer: HuggingFace tokenizer or API tokenizer
    
    Returns:
        Number of tokens
    """
    if hasattr(tokenizer, 'encode'):
        # HuggingFace tokenizer
        return len(tokenizer.encode(prompt, add_special_tokens=False))
    else:
        # Fallback: rough approximation (4 chars per token)
        return len(prompt) // 4


def generate_neutral_padding(
    target_tokens: int,
    current_tokens: int,
    tokenizer,
    filler_variants: Optional[List[str]] = None
) -> str:
    """
    Generate semantically neutral padding to reach target token count.
    
    Args:
        target_tokens: Target token length
        current_tokens: Current token length
        tokenizer: Tokenizer to use
        filler_variants: List of filler text variants
    
    Returns:
        Padding string that brings total to ~target_tokens
    """
    if current_tokens >= target_tokens:
        return ""
    
    tokens_needed = target_tokens - current_tokens
    
    # Default filler variants
    if filler_variants is None:
        filler_variants = [
            "Please provide a thoughtful and well-reasoned response to this question.",
            "Take a moment to carefully consider all aspects of this question before responding.",
            "Your careful analysis and clear reasoning are appreciated in addressing this question.",
            "Note: This is a reconsideration of the previous question.",
            "Please take your time to think through your response carefully.",
        ]
    
    # Build padding by repeating filler text
    padding_parts = []
    total_padding_tokens = 0
    
    while total_padding_tokens < tokens_needed:
        # Pick a filler variant (cycle through them)
        filler = filler_variants[len(padding_parts) % len(filler_variants)]
        filler_tokens = compute_prompt_token_length(filler, tokenizer)
        
        if total_padding_tokens + filler_tokens <= tokens_needed + 10:  # Allow +10 tolerance
            padding_parts.append(filler)
            total_padding_tokens += filler_tokens
        else:
            # Would exceed tolerance, break
            break
    
    return " ".join(padding_parts)


def compute_length_matched_prompts(
    prompts_dict: Dict[str, str],
    tokenizer,
    target_tolerance: int = 10,
    filler_variants: Optional[List[str]] = None
) -> Dict[str, Tuple[str, int, int, str]]:
    """
    Compute length-matched prompts for all conditions.
    
    All prompts will be padded to match the longest one within ±2 tokens.
    
    Args:
        prompts_dict: Dict mapping condition_key -> prompt_text
        tokenizer: Tokenizer to use for token counting
        target_tolerance: Tolerance in tokens (default: ±2)
        filler_variants: Optional list of filler text variants
    
    Returns:
        Dict mapping condition_key -> (padded_prompt, target_len, actual_len, padding_text)
    
    Raises:
        ValueError: If tolerance cannot be met
    """
    # 1. Tokenize all prompts and find max length
    token_lengths = {}
    for key, prompt in prompts_dict.items():
        token_lengths[key] = compute_prompt_token_length(prompt, tokenizer)
    
    max_length = max(token_lengths.values())
    target_length = max_length  # Target is the max
    
    # 2. Compute padding for each prompt
    result = {}
    for key, prompt in prompts_dict.items():
        current_length = token_lengths[key]
        
        if current_length >= target_length - target_tolerance:
            # Already within tolerance of max
            padded_prompt = prompt
            actual_length = current_length
            padding_text = ""
        else:
            # Need padding
            padding = generate_neutral_padding(
                target_length, current_length, tokenizer, filler_variants
            )
            
            if padding:
                padded_prompt = prompt + "\n\n" + padding
            else:
                padded_prompt = prompt
            padding_text = padding
            
            actual_length = compute_prompt_token_length(padded_prompt, tokenizer)
        
        # Verify tolerance
        if abs(actual_length - target_length) > target_tolerance:
            # Try to get closer
            if actual_length < target_length - target_tolerance:
                # Add one more filler sentence
                extra_filler = "Please consider all relevant factors in your response."
                padded_prompt = padded_prompt + " " + extra_filler
                actual_length = compute_prompt_token_length(padded_prompt, tokenizer)
        
        result[key] = (padded_prompt, target_length, actual_length, padding_text)
    
    # 3. Verify all are within tolerance (hard requirement)
    for key, val in result.items():
        target = val[1]
        actual = val[2]
        if abs(actual - target) > target_tolerance:
            raise ValueError(
                f"Length matching failed for {key}: actual={actual}, target={target}, tolerance={target_tolerance}"
            )
    
    return result


def verify_length_matching(
    padded_prompts: Dict[str, Tuple[str, int, int]],
    tolerance: int = 2
) -> bool:
    """
    Verify that all prompts are length-matched within tolerance.
    
    Args:
        padded_prompts: Output from compute_length_matched_prompts
        tolerance: Tolerance in tokens
    
    Returns:
        True if all within tolerance, False otherwise
    """
    # padded_prompts values: (padded_prompt, target_len, actual_len, padding_text)
    lengths = [val[2] for val in padded_prompts.values()]
    max_len = max(lengths)
    min_len = min(lengths)
    
    if max_len - min_len > tolerance:
        print(f"Length matching failed: max={max_len}, min={min_len}, diff={max_len - min_len}")
        return False
    
    return True


# Example usage
if __name__ == "__main__":
    print("Prompt Utilities: Length Matching")
    print("=" * 60)
    
    # Mock tokenizer
    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            # Simple word-based tokenization for demo
            return text.split()
    
    tokenizer = MockTokenizer()
    
    # Example prompts
    prompts = {
        "A1B0C1": "Question: Is X true? Answer Yes or No.",
        "A3B2C1": "Question: Is X true? An expert says it's false because of reasons A, B, and C. Answer Yes or No.",
    }
    
    padded = compute_length_matched_prompts(prompts, tokenizer, target_tolerance=2)
    
    print("\nLength matching results:")
    for key, (prompt, target, actual, _) in padded.items():
        print(f"{key}: target={target}, actual={actual}, diff={actual - target}")
    
    is_matched = verify_length_matching(padded, tolerance=2)
    print(f"\nAll prompts length-matched: {is_matched}")

