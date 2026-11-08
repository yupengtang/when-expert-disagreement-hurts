#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main audit: two-pass revision experiment on the 1,989-item benchmark.

For each question and model, Pass 1 is sampled once and reused as the shared
initial answer for every Pass-2 condition. The reported configuration
(configs/main_audit.json) runs A0 (neutral reflection), A1 (anonymous
disagreement) and A3 (expert-attributed disagreement) at T=0.7 with a 600-token
limit and one replicate. API models are queried in parallel through OpenRouter.
Local models run sequentially, are loaded in 8-bit precision when bitsandbytes
is available on a CUDA device, and also record teacher-forced Yes/No log-odds
in both passes.

Example, from the experiment/ directory:
    python run_experiment.py --models-to-run openai/gpt-4o-mini
"""

import os

# Set HuggingFace cache to a writable project directory by default.
HF_HOME = os.environ.get(
    "HF_HOME",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache"),
)
os.environ["HF_HOME"] = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "transformers")
os.makedirs(HF_HOME, exist_ok=True)

import json
import argparse
import hashlib
from pathlib import Path
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(path: str = ".env") -> None:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"").strip("'"))
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Load environment variables
load_dotenv()

from src.data_structures import (
    ExperimentConfig, Question, PassResult, TrialRecord,
    load_questions, write_jsonl_record, generate_trial_id, get_commit_hash
)
from src.experiment_runner import TwoPassExperiment
from src.prompt_utils import compute_length_matched_prompts, verify_length_matching
from src.aep_transformation import should_apply_aep, transform_question_to_aep, AEPMetadata

try:
    from src.mechanistic_probing import load_model_for_probing
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from client import Client

HERE = Path(__file__).resolve().parent
ARTIFACT = HERE.parent


# ============================================================
# Global thread-safe output lock (shared across all workers)
# ============================================================
_output_lock = threading.Lock()


def load_models_config(config_path: Path) -> dict:
    """Load frozen models configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_prompts_registry(registry_path: Path) -> dict:
    """Load frozen prompts registry"""
    with open(registry_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# THREAD-SAFE WORKER FUNCTION (following reference project pattern)
# ============================================================

def run_single_trial(args_tuple: Tuple) -> Tuple[int, int, int, int]:
    """
    Run a single trial (one question × one model × all temps × all replicates × all conditions).
    
    This function is designed for thread-safe parallel execution:
    - Creates its own Client() instance per call
    - Creates its own TwoPassExperiment instance per call
    - No shared mutable state except the output lock
    
    Args (passed as tuple for ThreadPoolExecutor compatibility):
        args_tuple: (question_dict, model_info, config_dict, prompts_registry, output_file_str, placebo_pools)
    
    Returns:
        Tuple of (pass1_success, pass1_invalid, trials_success, trials_error)
    """
    (question_dict, model_info, config_dict, prompts_registry, output_file_str, placebo_pools) = args_tuple
    
    # Reconstruct objects from serializable data
    question = Question(**question_dict)
    config = ExperimentConfig(**config_dict)
    output_file = Path(output_file_str)
    
    # Create thread-local instances (KEY PATTERN from reference project)
    client = Client()  # Each thread gets its own client
    experiment = TwoPassExperiment(config, prompts_registry, _output_lock)
    experiment.placebo_pools = placebo_pools
    
    # Stats for this task
    p1_success = 0
    p1_invalid = 0
    t_success = 0
    t_error = 0
    
    model_id = model_info["model_id"]
    provider = model_info["provider"]
    
    # Pre-compute conditions
    conditions = list(product(
        config.reputation_levels,
        config.evidence_levels,
        config.framing_levels
    ))
    
    for temperature in config.core_temperatures:
        for replicate_id in range(1, config.core_replicates + 1):
            
            # ===== PASS 1: RUN ONCE =====
            pass1_prompt = experiment.build_pass1_prompt(question)
            
            if provider == "local":
                # Skip local models in parallel mode (GPU not thread-safe)
                p1_invalid += 1
                continue
            else:
                try:
                    pass1_result = experiment.run_pass_api(
                        client, model_id, pass1_prompt, temperature, config.core_top_p, config.core_max_tokens
                    )
                except Exception as e:
                    # API error on Pass 1
                    p1_invalid += 1
                    continue
            
            # Validate Pass 1 result
            if pass1_result is None or pass1_result.parsed_label is None or \
               pass1_result.truncated or pass1_result.format_violation:
                p1_invalid += 1
                continue
            
            p1_success += 1
            
            # ===== PASS 2: RUN FOR ALL CONDITIONS (using shared Pass 1) =====
            
            # Build all Pass 2 prompts for length matching (cache per question)
            all_conditions = list(product(
                config.reputation_levels,
                config.evidence_levels,
                config.framing_levels
            ))
            
            prompts_dict = {}
            for (rep, evid, frm) in all_conditions:
                key = f"{rep}_{evid}_{frm}"
                prompt = experiment.build_pass2_prompt(
                    question, pass1_result.parsed_label, rep, evid, frm, padding=""
                )
                prompts_dict[key] = prompt
            
            # Compute length-matched prompts (API: use reference tokenizer)
            try:
                from transformers import AutoTokenizer
                ref_tok_id = getattr(config, "api_reference_tokenizer_model_id", "gpt2")
                tokenizer = AutoTokenizer.from_pretrained(ref_tok_id)
            except Exception:
                tokenizer = None

            if tokenizer is not None:
                filler_variants = prompts_registry.get("padding_policy", {}).get("filler_variants", None)
                padded_prompts = compute_length_matched_prompts(
                    prompts_dict, tokenizer, target_tolerance=10, filler_variants=filler_variants
                )
            else:
                padded_prompts = {k: (v, 0, 0, "") for k, v in prompts_dict.items()}
            
            # Run Pass 2 for each condition
            for condition in conditions:
                reputation_level, evidence_level, framing_level = condition
                condition_key = f"{reputation_level}_{evidence_level}_{framing_level}"
                
                try:
                    pass2_prompt, target_len, actual_len, padding_text = padded_prompts[condition_key]
                    
                    pass2_result = experiment.run_pass_api(
                        client, model_id, pass2_prompt, temperature, config.core_top_p, config.core_max_tokens
                    )
                    
                    # Compute reversal
                    reversal = (
                        pass1_result.parsed_label is not None and
                        pass2_result.parsed_label is not None and
                        pass1_result.parsed_label != pass2_result.parsed_label
                    )
                    
                    # Compute delta log-odds (API models typically don't have this)
                    delta_logodds = None
                    if pass1_result.logodds_yes_over_no is not None and pass2_result.logodds_yes_over_no is not None:
                        delta_logodds = pass2_result.logodds_yes_over_no - pass1_result.logodds_yes_over_no
                    
                    # Generate trial ID
                    trial_id = generate_trial_id(model_id, question.question_id, condition_key, replicate_id)
                    
                    # Create trial record
                    record = TrialRecord(
                        trial_id=trial_id,
                        question_id=question.question_id,
                        domain=question.domain,
                        question_hash=question.hash,
                        excerpt_hash=hashlib.sha256(question.excerpt.encode("utf-8")).hexdigest()[:16],
                        model_id=model_id,
                        provider=provider,
                        timestamp_utc=datetime.utcnow().isoformat(),
                        commit_hash=get_commit_hash(),
                        temperature=temperature,
                        top_p=config.core_top_p,
                        max_tokens=config.core_max_tokens,
                        stop_sequences=experiment.stop_sequences,
                        replicate_id=replicate_id,
                        reputation_factor=reputation_level,
                        evidence_factor=evidence_level,
                        framing_factor=framing_level,
                        pass1=pass1_result,
                        pass2=pass2_result,
                        reversal=reversal,
                        delta_logodds=delta_logodds,
                        target_token_len=target_len,
                        actual_token_len=actual_len,
                        padding_id=hashlib.sha256((padding_text or "").encode("utf-8")).hexdigest()[:8],
                        addon_hash=getattr(experiment, "_last_addon_hash", ""),
                        addon_source_id=getattr(experiment, "_last_addon_source_id", "")
                    )
                    
                    # Write to file (thread-safe)
                    write_jsonl_record(record, output_file, lock=_output_lock)
                    t_success += 1
                    
                except Exception as e:
                    t_error += 1
    
    return (p1_success, p1_invalid, t_success, t_error)


# ============================================================
# SEQUENTIAL RUNNER (for local GPU models - not thread-safe)
# ============================================================

def run_local_model_sequential(
    questions: List[Question],
    model_info: dict,
    config: ExperimentConfig,
    prompts_registry: dict,
    output_file: Path,
    placebo_pools: dict,
    device: str = "cuda"
) -> Tuple[int, int, int, int]:
    """
    Run experiments for a local GPU model sequentially.
    GPU models are NOT thread-safe, so we run them one at a time.
    """
    experiment = TwoPassExperiment(config, prompts_registry, _output_lock)
    experiment.placebo_pools = placebo_pools
    
    model_id = model_info["model_id"]
    
    # Load model once and reuse
    if not TORCH_AVAILABLE:
        print(f"Note: torch not available; skipping local model {model_id}")
        return (0, 0, 0, 0)
    
    model, tokenizer = load_model_for_probing(model_id, device=device)
    
    p1_success = 0
    p1_invalid = 0
    t_success = 0
    t_error = 0
    
    conditions = list(product(
        config.reputation_levels,
        config.evidence_levels,
        config.framing_levels
    ))
    
    total_questions = len(questions)
    
    for q_idx, question in enumerate(questions):
        print(f"\r  Processing question {q_idx+1}/{total_questions}...", end="", flush=True)
        
        for temperature in config.core_temperatures:
            for replicate_id in range(1, config.core_replicates + 1):
                
                # Pass 1
                pass1_prompt = experiment.build_pass1_prompt(question)
                pass1_result = experiment.run_pass_local(
                    model, tokenizer, pass1_prompt, temperature, config.core_top_p, config.core_max_tokens, device
                )
                
                if pass1_result is None or pass1_result.parsed_label is None or \
                   pass1_result.truncated or pass1_result.format_violation:
                    p1_invalid += 1
                    continue
                
                p1_success += 1
                
                # Build length-matched prompts
                prompts_dict = {}
                for (rep, evid, frm) in conditions:
                    key = f"{rep}_{evid}_{frm}"
                    prompt = experiment.build_pass2_prompt(
                        question, pass1_result.parsed_label, rep, evid, frm, padding=""
                    )
                    prompts_dict[key] = prompt
                
                filler_variants = prompts_registry.get("padding_policy", {}).get("filler_variants", None)
                padded_prompts = compute_length_matched_prompts(
                    prompts_dict, tokenizer, target_tolerance=10, filler_variants=filler_variants
                )
                
                # Pass 2 for all conditions
                for condition in conditions:
                    reputation_level, evidence_level, framing_level = condition
                    condition_key = f"{reputation_level}_{evidence_level}_{framing_level}"
                    
                    try:
                        pass2_prompt, target_len, actual_len, padding_text = padded_prompts[condition_key]
                        
                        pass2_result = experiment.run_pass_local(
                            model, tokenizer, pass2_prompt, temperature, config.core_top_p, config.core_max_tokens, device
                        )
                        
                        reversal = (
                            pass1_result.parsed_label is not None and
                            pass2_result.parsed_label is not None and
                            pass1_result.parsed_label != pass2_result.parsed_label
                        )
                        
                        delta_logodds = None
                        if pass1_result.logodds_yes_over_no is not None and pass2_result.logodds_yes_over_no is not None:
                            delta_logodds = pass2_result.logodds_yes_over_no - pass1_result.logodds_yes_over_no
                        
                        trial_id = generate_trial_id(model_id, question.question_id, condition_key, replicate_id)
                        
                        record = TrialRecord(
                            trial_id=trial_id,
                            question_id=question.question_id,
                            domain=question.domain,
                            question_hash=question.hash,
                            excerpt_hash=hashlib.sha256(question.excerpt.encode("utf-8")).hexdigest()[:16],
                            model_id=model_id,
                            provider="local",
                            timestamp_utc=datetime.utcnow().isoformat(),
                            commit_hash=get_commit_hash(),
                            temperature=temperature,
                            top_p=config.core_top_p,
                            max_tokens=config.core_max_tokens,
                            stop_sequences=experiment.stop_sequences,
                            replicate_id=replicate_id,
                            reputation_factor=reputation_level,
                            evidence_factor=evidence_level,
                            framing_factor=framing_level,
                            pass1=pass1_result,
                            pass2=pass2_result,
                            reversal=reversal,
                            delta_logodds=delta_logodds,
                            target_token_len=target_len,
                            actual_token_len=actual_len,
                            padding_id=hashlib.sha256((padding_text or "").encode("utf-8")).hexdigest()[:8],
                            addon_hash=getattr(experiment, "_last_addon_hash", ""),
                            addon_source_id=getattr(experiment, "_last_addon_source_id", "")
                        )
                        
                        write_jsonl_record(record, output_file, lock=_output_lock)
                        t_success += 1
                        
                    except Exception as e:
                        t_error += 1
    
    print()  # Newline after progress
    
    # Cleanup GPU memory
    del model, tokenizer
    if TORCH_AVAILABLE:
        import torch
        torch.cuda.empty_cache()
    
    return (p1_success, p1_invalid, t_success, t_error)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run the two-pass main audit")
    parser.add_argument("--config", type=str, default=str(HERE / "configs" / "main_audit.json"))
    parser.add_argument("--questions", type=str,
                        default=str(ARTIFACT / "data-repair" / "outputs" / "original_benchmark_recovered.jsonl"))
    parser.add_argument("--prompts", type=str, default=str(HERE / "prompts" / "registry.json"))
    parser.add_argument("--models", type=str, default=str(HERE / "models.json"))
    parser.add_argument("--output", type=str, default=str(ARTIFACT / "runs" / "main_audit_trials.jsonl"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--models-to-run", type=str, default=None, help="Comma-separated list of model IDs to run")
    # Parallel control (can be overridden by config)
    parser.add_argument("--parallel", action="store_true", default=None, help="Enable parallel processing")
    parser.add_argument("--no-parallel", dest="parallel", action="store_false", help="Disable parallel processing")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum number of parallel workers")
    
    args = parser.parse_args()
    
    # Load configuration
    config = ExperimentConfig.load(Path(args.config))
    
    # Command-line args override config
    if args.parallel is not None:
        config.parallel = args.parallel
    if args.max_workers is not None:
        config.max_workers = args.max_workers
    
    # Test mode
    if args.test_mode:
        print("\n" + "="*60)
        print("TEST MODE")
        print("="*60)
        config.core_temperatures = [0.0]
        config.core_replicates = 1
        config.reputation_levels = ["A0", "A1"]
        config.evidence_levels = ["B0"]
        config.framing_levels = ["C1"]
    
    # Load resources
    print("Loading resources...")
    questions = load_questions(Path(args.questions))
    
    # Apply AEP transformation if enabled
    if config.ablations.get("aep_entity_virtualization", False):
        aep_ratio = config.ablations.get("aep_ratio", 0.5)
        aep_count = 0
        
        for i, q in enumerate(questions):
            if should_apply_aep(q.question_id, aep_ratio):
                transformed_prompt, was_t_prompt, replaced_p = transform_question_to_aep(q.prompt_base)
                transformed_excerpt, was_t_excerpt, replaced_e = transform_question_to_aep(q.excerpt)

                replaced = {**replaced_p, **replaced_e}
                was_transformed = was_t_prompt or was_t_excerpt
                if was_transformed:
                    q.prompt_base = transformed_prompt
                    q.excerpt = transformed_excerpt
                    q.is_aep = True
                    q.metadata["aep_entities_replaced"] = replaced
                    aep_count += 1
        
        print(f"Applied AEP to {aep_count}/{len(questions)} questions ({aep_count/len(questions)*100:.1f}%)")
    
    if args.test_mode:
        questions = questions[:1]
    print(f"Loaded {len(questions)} questions")
    
    prompts_registry = load_prompts_registry(Path(args.prompts))
    models_config = load_models_config(Path(args.models))
    
    # Select models
    models_to_run = []
    
    if args.models_to_run:
        target_ids = [m_id.strip() for m_id in args.models_to_run.split(",")]
        all_models = models_config.get("mechanistic_probing_set", {}).get("core", []) + \
                     models_config.get("behavioral_anchor_set", {}).get("models", [])
        models_to_run = [m for m in all_models if m["model_id"] in target_ids]
    else:
        if config.run_mechanistic_core and TORCH_AVAILABLE:
            models_to_run.extend(models_config["mechanistic_probing_set"]["core"])
        elif config.run_mechanistic_core and not TORCH_AVAILABLE:
            print("Note: torch not available; skipping local mechanistic models.")
        if config.run_behavioral_anchors:
            models_to_run.extend(models_config["behavioral_anchor_set"]["models"])
    
    print(f"Running {len(models_to_run)} model(s): {[m['model_id'] for m in models_to_run]}")
    
    # Separate local and API models
    local_models = [m for m in models_to_run if m.get("provider") == "local"]
    api_models = [m for m in models_to_run if m.get("provider") != "local"]
    
    # Generate conditions
    conditions = list(product(
        config.reputation_levels,
        config.evidence_levels,
        config.framing_levels
    ))
    print(f"Testing {len(conditions)} condition(s)")
    
    # Output setup
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Build placebo pools
    placebo_pools = {}
    for q in questions:
        b2 = (q.metadata or {}).get("b2_evidence")
        if isinstance(b2, str) and b2.strip():
            placebo_pools.setdefault(q.domain, []).append((q.question_id, b2.strip()))
    
    # Calculate totals
    total_pass1_local = len(questions) * len(local_models) * len(config.core_temperatures) * config.core_replicates
    total_pass1_api = len(questions) * len(api_models) * len(config.core_temperatures) * config.core_replicates
    total_pass2_local = total_pass1_local * len(conditions)
    total_pass2_api = total_pass1_api * len(conditions)
    
    print(f"\n{'='*60}")
    print("STARTING EXPERIMENT (V3 - Robust Parallel)")
    if api_models and config.parallel and config.max_workers > 1:
        print(f"API MODELS: PARALLEL MODE with {config.max_workers} workers")
    elif api_models:
        print("API MODELS: SEQUENTIAL MODE")
    if local_models:
        print("LOCAL MODELS: SEQUENTIAL MODE (GPU not thread-safe)")
    print(f"{'='*60}\n")
    
    print(f"Local models: {len(local_models)} (trials: {total_pass2_local})")
    print(f"API models: {len(api_models)} (trials: {total_pass2_api})")
    print(f"Total trials: {total_pass2_local + total_pass2_api}\n")
    
    # Aggregated stats
    total_p1_success = 0
    total_p1_invalid = 0
    total_t_success = 0
    total_t_error = 0
    
    # ===== RUN LOCAL MODELS (Sequential) =====
    if local_models:
        print(f"\n--- Processing {len(local_models)} local model(s) sequentially ---")
        for model_info in local_models:
            print(f"\nRunning local model: {model_info['model_id']}")
            p1_s, p1_i, t_s, t_e = run_local_model_sequential(
                questions, model_info, config, prompts_registry, output_file, placebo_pools, args.device
            )
            total_p1_success += p1_s
            total_p1_invalid += p1_i
            total_t_success += t_s
            total_t_error += t_e
            print(f"  Completed: {t_s} trials, {t_e} errors")
    
    # ===== RUN API MODELS (Parallel or Sequential) =====
    if api_models:
        print(f"\n--- Processing {len(api_models)} API model(s) ---")
        
        # Prepare tasks: each task = (question, model) pair
        # Convert Question objects to dicts for thread-safety
        config_dict = config.to_dict()
        output_file_str = str(output_file)
        
        tasks = []
        for question in questions:
            for model_info in api_models:
                task_tuple = (
                    question.to_dict(),  # Serialize Question to dict
                    model_info,
                    config_dict,
                    prompts_registry,
                    output_file_str,
                    placebo_pools
                )
                tasks.append(task_tuple)
        
        print(f"Prepared {len(tasks)} task(s) for API models")
        
        if config.parallel and config.max_workers > 1:
            # Parallel execution
            print(f"Running in parallel with {config.max_workers} workers...")
            
            with tqdm(total=len(tasks), desc="Running API trials") as pbar:
                with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                    # Submit all tasks
                    future_to_task = {executor.submit(run_single_trial, task): task for task in tasks}
                    
                    # Process completed tasks
                    for future in as_completed(future_to_task):
                        try:
                            p1_s, p1_i, t_s, t_e = future.result()
                            total_p1_success += p1_s
                            total_p1_invalid += p1_i
                            total_t_success += t_s
                            total_t_error += t_e
                        except Exception as exc:
                            print(f"\nTask generated an exception: {exc}")
                            total_t_error += 1
                        
                        pbar.update(1)
        else:
            # Sequential execution
            print("Running sequentially...")
            
            with tqdm(total=len(tasks), desc="Running API trials") as pbar:
                for task in tasks:
                    try:
                        p1_s, p1_i, t_s, t_e = run_single_trial(task)
                        total_p1_success += p1_s
                        total_p1_invalid += p1_i
                        total_t_success += t_s
                        total_t_error += t_e
                    except Exception as exc:
                        print(f"\nTask generated an exception: {exc}")
                        total_t_error += 1
                    
                    pbar.update(1)
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Pass 1 runs:")
    print(f"  Success: {total_p1_success}")
    print(f"  Invalid: {total_p1_invalid}")
    print(f"Trials (Pass 2):")
    print(f"  Success: {total_t_success}")
    print(f"  Errors: {total_t_error}")
    print(f"Output: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
