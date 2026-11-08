#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-Pass Experiment Protocol for BEAT-300

This module implements the core experimental protocol:
- Pass 1: Baseline answer
- Pass 2: Intervention (reputation × evidence × framing)

Supports both local models (with mechanistic probing) and API models (behavioral only).
"""

import json
import hashlib
import time
import threading
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from datetime import datetime
import re

try:
    import torch
    from src.mechanistic_probing import yes_no_logodds, load_model_for_probing
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.data_structures import (
    Question, PassResult, TrialRecord, ExperimentConfig,
    generate_trial_id, write_jsonl_record, get_commit_hash
)
from client import Client  # OpenRouter API client


class TwoPassExperiment:
    """Core experimental protocol: Pass 1 (baseline) + Pass 2 (intervention)"""
    
    def __init__(
        self,
        config: ExperimentConfig,
        prompts_registry: Dict[str, Any],
        output_lock: Optional[threading.Lock] = None
    ):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
            prompts_registry: Loaded prompts registry (A/B/C factors)
            output_lock: Optional lock for thread-safe file writes
        """
        self.config = config
        self.prompts_registry = prompts_registry
        self.output_lock = output_lock or threading.Lock()
        
        # Load stop sequences from registry
        self.stop_sequences = prompts_registry.get("output_contract", {}).get(
            "stop_sequences", 
            ["</s>", "###", "User:", "Assistant:", "Q:", "A:", "B:"]
        )
        # Optional: pool for B3 placebo add-ons (domain -> list[(question_id, addon_text)])
        self.placebo_pools: Dict[str, List[Tuple[str, str]]] = {}

    @staticmethod
    def _hash16(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def build_question_block(self, question: Question) -> str:
        """
        Build the invariant question block (excerpt + prompt_base).
        This block is identical across Pass 1 and Pass 2.
        """
        return f"Evidence Excerpt:\n\n{question.excerpt}\n\n---\n\n{question.prompt_base}"
    
    def build_pass1_prompt(self, question: Question) -> str:
        """Build Pass 1 prompt (baseline, no intervention)"""
        return self.build_question_block(question)
    
    def build_pass2_prompt(
        self,
        question: Question,
        pass1_answer: str,
        reputation_level: str,
        evidence_level: str,
        framing_level: str,
        padding: str = "",
        disagreement_rate: float = 1.0,
        force_disagree: bool = True
    ) -> str:
        """
        Build Pass 2 prompt with condition intervention.
        
        Args:
            question: Question object
            pass1_answer: Pass 1 answer ("Yes" or "No")
            reputation_level: A0, A1, A2, A3, A3_fallible
            evidence_level: B0, B1, B2, B3
            framing_level: C1, C2
            padding: Length-matching padding
            disagreement_rate: Probability that opponent disagrees (default: 1.0 = always)
            force_disagree: If True, ignore disagreement_rate (for main experiment)
        
        Returns:
            Formatted Pass 2 prompt
        """
        rep_cfg = self.prompts_registry["reputation_factor"][reputation_level]
        framing_cfg = self.prompts_registry["framing_factor"][framing_level]
        evid_cfg = self.prompts_registry["evidence_factor"][evidence_level]
        
        # Determine opponent answer based on disagreement rate
        if force_disagree or (disagreement_rate >= 1.0):
            # Max-contrast: always opposite (main experiment)
            opponent_answer = "No" if pass1_answer == "Yes" else "Yes"
        else:
            # Partial-disagreement ablation
            import random
            if random.random() < disagreement_rate:
                opponent_answer = "No" if pass1_answer == "Yes" else "Yes"
            else:
                opponent_answer = pass1_answer  # Agree

        domain = question.domain

        # Reputation block (evidence-free)
        rep_block = rep_cfg["template"].format(
            opponent_answer=opponent_answer,
            domain=domain
        ).strip()

        # Anti-confound: enforce non-evaluative, evidence-free reputation block
        for forbidden in rep_cfg.get("blacklist", []):
            if forbidden.lower() in rep_block.lower():
                raise ValueError(f"Reputation block violates blacklist term: {forbidden}")
        for pat in self.prompts_registry.get("opponent_minimality_policy", {}).get("forbidden_regex", []):
            if re.search(pat, rep_block, flags=re.IGNORECASE):
                raise ValueError(f"Reputation block violates forbidden pattern: {pat}")

        # Evidence add-on block
        addon_text = ""
        addon_source_id = ""
        if evidence_level == "B1":
            addon_text = evid_cfg["template"].strip()
        elif evidence_level == "B2":
            # Dataset-native strong relevant add-on
            b2 = (question.metadata or {}).get("b2_evidence") or ""
            if not isinstance(b2, str) or not b2.strip():
                # If missing, still include an empty block to avoid leaking evidence.
                b2 = ""
            if domain.lower() == "medicine":
                tpl = evid_cfg["template_medicine"]
                addon_text = tpl.format(long_answer=b2.strip()).strip()
            elif domain.lower() == "science":
                tpl = evid_cfg["template_science"]
                addon_text = tpl.format(rationale_sentences=b2.strip()).strip()
            else:
                tpl = evid_cfg["template_law"]
                addon_text = tpl.format(evidence_spans=b2.strip()).strip()
        elif evidence_level == "B3":
            # Placebo add-on: unrelated item within same domain, logged via addon_source_id
            pool = self.placebo_pools.get(domain, [])
            placebo_content = ""
            if pool:
                # Pick first content that is not from same question_id if present in pool tuples
                for sid, txt in pool:
                    if sid != question.question_id and isinstance(txt, str) and txt.strip():
                        addon_source_id = sid
                        placebo_content = txt.strip()
                        break
                if not placebo_content:
                    addon_source_id, placebo_content = pool[0][0], (pool[0][1] or "")
            if domain.lower() == "medicine":
                tpl = evid_cfg["template_medicine"]
                addon_text = tpl.format(placebo_content=placebo_content).strip()
            elif domain.lower() == "science":
                tpl = evid_cfg["template_science"]
                addon_text = tpl.format(placebo_content=placebo_content).strip()
            else:
                tpl = evid_cfg["template_law"]
                addon_text = tpl.format(placebo_content=placebo_content).strip()

        # Framing block
        framing_block = framing_cfg["template"].strip()

        # Compose context block (only this part varies across conditions; question block invariant)
        context_parts = [p for p in [framing_block, rep_block, addon_text] if p]
        context_block = "\n\n".join(context_parts).strip()

        question_block = self.build_question_block(question)

        full = f"{context_block}\n\n{question_block}" if context_block else question_block

        if padding:
            full = full + "\n\n" + padding

        # Attach add-on bookkeeping for caller (stored on instance for retrieval)
        self._last_addon_hash = self._hash16(addon_text) if addon_text else ""
        self._last_addon_source_id = addon_source_id

        return full
    
    def parse_response(self, response: str) -> Tuple[Optional[str], str, bool, bool]:
        """
        Parse model response.
        
        Returns:
            Tuple of (answer, reasoning, truncated, format_violation)
            - answer: "Yes" or "No" or None
            - reasoning: Extracted reasoning text
            - truncated: Whether response appears truncated
            - format_violation: Whether format is invalid
        """
        import re
        
        response = response.strip()
        
        # Check for refusal
        refusal_patterns = [
            r"I (?:cannot|can't|shouldn't) (?:answer|provide|make)",
            r"(?:unable|impossible) to (?:answer|determine|decide)",
            r"I don't have (?:enough|sufficient) information"
        ]
        
        response_lower = response.lower()
        for pattern in refusal_patterns:
            if re.search(pattern, response_lower):
                return None, response, False, True
        
        # First try: Check if first line is exactly Yes or No (Llama format)
        lines = response.split('\n')
        first_line = lines[0].strip().upper()
        
        answer = None
        reasoning = ""
        
        if first_line == 'YES':
            answer = "Yes"
            reasoning = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""
        elif first_line == 'NO':
            answer = "No"
            reasoning = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""
        else:
            # Second try: Check if response starts with "Yes," or "No," (Mistral format)
            if response.startswith('Yes, ') or response.startswith('Yes.'):
                answer = "Yes"
                # Extract everything after "Yes, " or "Yes. "
                reasoning = response[5:].strip() if response[3:5] == ', ' else response[4:].strip()
            elif response.startswith('No, ') or response.startswith('No.'):
                answer = "No"
                reasoning = response[4:].strip() if response[2:4] == ', ' else response[3:].strip()
        
        # Check format violation
        format_violation = (answer is None)
        
        # Check truncation: reasoning should end with punctuation or be reasonably long
        truncated = False
        if reasoning:
            # Ultra-relaxed check: if it's longer than 20 chars, assume it's valid
            # We don't care about perfect punctuation for this experiment
            if len(reasoning) < 20: 
                truncated = True
        else:
            # No reasoning provided
            truncated = True
        
        return answer, reasoning, truncated, format_violation
    
    def run_pass_local(
        self,
        model,
        tokenizer,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        device: str = "cuda",
        compute_logodds: bool = True
    ) -> PassResult:
        """
        Run a single pass on a local model.
        
        Args:
            model: Loaded HuggingFace model
            tokenizer: Tokenizer
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Max generation tokens
            device: Device
            compute_logodds: Whether to compute mechanistic log-odds
        
        Returns:
            PassResult with response and optionally log-odds
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch not available for local model execution")
        
        # Format prompt with chat template
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, 'apply_chat_template'):
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        # Generate
        start_time = time.time()
        
        # Prepare stop sequences for local generation
        # Convert string stop sequences to token IDs if possible
        stop_token_ids = []
        if hasattr(tokenizer, 'encode'):
            for stop_str in self.stop_sequences:
                try:
                    # Try to encode each stop sequence
                    tokens = tokenizer.encode(stop_str, add_special_tokens=False)
                    if tokens:
                        stop_token_ids.extend(tokens)
                except:
                    pass
        
        # Add EOS token
        if tokenizer.eos_token_id is not None:
            stop_token_ids.append(tokenizer.eos_token_id)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                top_p=top_p,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=stop_token_ids if stop_token_ids else tokenizer.eos_token_id
            )
        latency = time.time() - start_time
        
        # Decode
        raw_output = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Parse
        answer, reasoning, truncated, format_violation = self.parse_response(raw_output)
        
        # Token usage (approximate for local)
        prompt_tokens = inputs['input_ids'].shape[1]
        completion_tokens = outputs.shape[1] - prompt_tokens
        
        # Mechanistic probing
        lp_yes, lp_no, logodds, variants_agg = None, None, None, None
        if compute_logodds and answer in ["Yes", "No"]:
            try:
                result = yes_no_logodds(model, tokenizer, formatted_prompt, device=device)
                lp_yes = result.lp_yes
                lp_no = result.lp_no
                logodds = result.log_odds_yes_over_no
                variants_agg = result.variants_aggregated
            except Exception as e:
                print(f"Warning: Log-odds computation failed: {e}")
        
        return PassResult(
            raw_output=raw_output,
            parsed_label=answer,
            reasoning=reasoning,
            truncated=truncated,
            format_violation=format_violation,
            token_usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            latency_seconds=latency,
            finish_reason="stop" if not truncated else "length",
            lp_yes=lp_yes,
            lp_no=lp_no,
            logodds_yes_over_no=logodds,
            variants_aggregated=variants_agg
        )
    
    def run_pass_api(
        self,
        client: Client,
        model_id: str,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int
    ) -> PassResult:
        """
        Run a single pass on an API model (OpenRouter).
        
        Args:
            client: OpenRouter client
            model_id: Model identifier
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Max generation tokens
        
        Returns:
            PassResult (no mechanistic probing)
        """
        messages = [{"role": "user", "content": prompt}]
        
        start_time = time.time()
        try:
            response = client.chat(
                model=model_id,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=self.stop_sequences
            )
            latency = time.time() - start_time
            
            raw_output = response.get("content", "")
            finish_reason = response.get("finish_reason", "unknown")
            usage = response.get("usage", {})
            
            # Parse
            answer, reasoning, truncated, format_violation = self.parse_response(raw_output)
            
            return PassResult(
                raw_output=raw_output,
                parsed_label=answer,
                reasoning=reasoning,
                truncated=(finish_reason == "length") or truncated,
                format_violation=format_violation,
                token_usage=usage,
                latency_seconds=latency,
                finish_reason=finish_reason
            )
        
        except Exception as e:
            latency = time.time() - start_time
            return PassResult(
                raw_output=f"ERROR: {str(e)}",
                parsed_label=None,
                reasoning="",
                truncated=False,
                format_violation=True,
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                latency_seconds=latency,
                finish_reason="error"
            )
    
    def run_trial(
        self,
        question: Question,
        model_id: str,
        provider: str,  # "local" or "openrouter"
        reputation_level: str,
        evidence_level: str,
        framing_level: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        replicate_id: int,
        model=None,  # For local models
        tokenizer=None,  # For local models
        client=None,  # For API models
        device: str = "cuda"
    ) -> Optional[TrialRecord]:
        """
        Run a complete trial (Pass 1 + Pass 2).
        
        Returns:
            TrialRecord or None if Pass 1 fails
        """
        # Generate trial ID
        condition_str = f"{reputation_level}_{evidence_level}_{framing_level}"
        trial_id = generate_trial_id(model_id, question.question_id, condition_str, replicate_id)
        
        # Pass 1: Baseline
        pass1_prompt = self.build_pass1_prompt(question)
        
        if provider == "local":
            if model is None or tokenizer is None:
                raise ValueError("Local model and tokenizer required for provider='local'")
            pass1_result = self.run_pass_local(
                model, tokenizer, pass1_prompt, temperature, top_p, max_tokens, device
            )
        else:
            if client is None:
                raise ValueError("Client required for API provider")
            pass1_result = self.run_pass_api(
                client, model_id, pass1_prompt, temperature, top_p, max_tokens
            )
        
        # Check if Pass 1 is valid
        if pass1_result.parsed_label is None or pass1_result.truncated or pass1_result.format_violation:
            # Invalid Pass 1 - skip trial
            return None
        
        # Pass 2: Intervention
        pass2_prompt = self.build_pass2_prompt(
            question,
            pass1_result.parsed_label,
            reputation_level,
            evidence_level,
            framing_level
        )
        
        if provider == "local":
            pass2_result = self.run_pass_local(
                model, tokenizer, pass2_prompt, temperature, top_p, max_tokens, device
            )
        else:
            pass2_result = self.run_pass_api(
                client, model_id, pass2_prompt, temperature, top_p, max_tokens
            )
        
        # Compute reversal
        reversal = (
            pass1_result.parsed_label is not None and
            pass2_result.parsed_label is not None and
            pass1_result.parsed_label != pass2_result.parsed_label
        )
        
        # Compute delta log-odds (mechanistic only)
        delta_logodds = None
        if pass1_result.logodds_yes_over_no is not None and pass2_result.logodds_yes_over_no is not None:
            delta_logodds = pass2_result.logodds_yes_over_no - pass1_result.logodds_yes_over_no
        
        # Create trial record
        excerpt_hash = self._hash16(question.excerpt)
        record = TrialRecord(
            trial_id=trial_id,
            question_id=question.question_id,
            domain=question.domain,
            question_hash=question.hash,
            excerpt_hash=excerpt_hash,
            model_id=model_id,
            provider=provider,
            timestamp_utc=datetime.utcnow().isoformat(),
            commit_hash=get_commit_hash(),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_sequences=self.stop_sequences,
            replicate_id=replicate_id,
            reputation_factor=reputation_level,
            evidence_factor=evidence_level,
            framing_factor=framing_level,
            pass1=pass1_result,
            pass2=pass2_result,
            reversal=reversal,
            delta_logodds=delta_logodds,
            addon_hash=getattr(self, "_last_addon_hash", ""),
            addon_source_id=getattr(self, "_last_addon_source_id", "")
        )
        
        return record


# Example usage
if __name__ == "__main__":
    print("Two-Pass Experiment Protocol")
    print("=" * 60)
    print("This module implements the core BEAT-300 experimental protocol:")
    print("- Pass 1: Baseline answer")
    print("- Pass 2: Intervention (A × B × C factors)")
    print("- Mechanistic probing for local models")
    print("- Behavioral measurement for all models")
    print("=" * 60)

