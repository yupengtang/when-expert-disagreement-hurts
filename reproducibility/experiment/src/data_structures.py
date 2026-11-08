#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Structures and Logging Schema for BEAT-300 Experiment
Defines all data structures, JSONL logging format, and utilities.
"""

import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime


@dataclass
class Question:
    """A benchmark question."""
    question_id: str
    domain: str  # "Medicine", "Science", "Law"
    prompt_base: str  # Base prompt without A/B/C interventions
    excerpt: str  # Native evidence excerpt (abstract, contract snippet, etc.)
    candidate_id: Optional[str] = field(default=None)
    source: Optional[str] = field(default=None)
    ground_truth: Optional[str] = field(default=None)
    confidence_tier: Optional[str] = field(default=None)
    yes_proportion_llama: Optional[float] = field(default=None)
    yes_proportion_mistral: Optional[float] = field(default=None)
    
    hash: str = field(default="")
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_aep: bool = False  # True if entities are virtualized (AEP transformation)
    excerpt_hash: Optional[str] = None
    input_sha256: Optional[str] = None
    dataset_version: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.hash:
            # Hash both prompt and excerpt for uniqueness
            combined = f"{self.prompt_base}|{self.excerpt}"
            self.hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PassResult:
    """Result from a single pass (Pass 1 or Pass 2)"""
    raw_output: str
    parsed_label: Optional[str]  # "Yes" or "No" or None
    reasoning: str
    truncated: bool
    format_violation: bool
    token_usage: Dict[str, int]  # {"prompt_tokens": X, "completion_tokens": Y, "total_tokens": Z}
    latency_seconds: float
    finish_reason: str  # "stop", "length", "error"
    
    # Mechanistic fields (only for open models)
    lp_yes: Optional[float] = None
    lp_no: Optional[float] = None
    logodds_yes_over_no: Optional[float] = None
    variants_aggregated: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrialRecord:
    """Complete record for a single trial (Pass 1 + Pass 2)"""
    # Identifiers
    trial_id: str
    question_id: str
    domain: str
    question_hash: str
    excerpt_hash: str  # Hash of native evidence excerpt
    
    # Run metadata
    model_id: str
    provider: str
    timestamp_utc: str
    commit_hash: str
    
    # Harness settings
    temperature: float
    top_p: float
    max_tokens: int
    stop_sequences: List[str]
    replicate_id: int  # Which replicate (1, 2, 3, ...)
    
    # Condition labels
    reputation_factor: str  # A0, A1, A2, A3
    evidence_factor: str  # B0, B1, B2, B3
    framing_factor: str  # C1, C2
    
    # Pass results
    pass1: PassResult
    pass2: PassResult
    
    # Derived outcomes
    reversal: bool  # Did textual answer change?
    delta_logodds: Optional[float] = None  # pass2_logodds - pass1_logodds (mechanistic only)
    
    # Padding metadata (for length matching validation)
    target_token_len: int = 0
    actual_token_len: int = 0
    padding_id: str = ""
    
    # Evidence add-on metadata (for B2/B3 provenance)
    addon_hash: str = ""
    addon_source_id: str = ""  # For B3 placebo tracking
    
    # Revision taxonomy (computed in analysis)
    revision_category: Optional[str] = None  # "Deep", "Superficial", "Latent", "Stable"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSONL serialization"""
        d = asdict(self)
        # Ensure timestamp is ISO format
        if not d['timestamp_utc']:
            d['timestamp_utc'] = datetime.utcnow().isoformat()
        return d
    
    def to_jsonl_line(self) -> str:
        """Serialize to a single JSONL line"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class ExperimentConfig:
    """Configuration for running the BEAT-300 experiment"""
    # Paths
    questions_file: str = "../data-repair/outputs/original_benchmark_recovered.jsonl"
    prompts_registry: str = "prompts/registry.json"
    models_config: str = "models.json"
    output_dir: str = "results"
    
    # Harness (core)
    core_temperatures: List[float] = field(default_factory=lambda: [0.0, 0.7])
    core_top_p: float = 1.0
    core_max_tokens: int = 120
    core_replicates: int = 3
    
    # Harness (extended, optional)
    extended_temperatures: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.7, 1.0])
    extended_top_p_values: List[float] = field(default_factory=lambda: [0.9, 1.0])
    extended_replicates: int = 5
    
    # Model sets to run
    run_mechanistic_core: bool = True
    run_mechanistic_extended: bool = False
    run_behavioral_anchors: bool = True
    run_reasoning_case_study: bool = False
    
    # Conditions to run (full factorial or subset)
    reputation_levels: List[str] = field(default_factory=lambda: ["A0", "A1", "A2", "A3"])
    reputation_levels_extended: List[str] = field(default_factory=lambda: ["A0", "A1", "A2", "A3", "A3_fallible"])
    evidence_levels: List[str] = field(default_factory=lambda: ["B0", "B1", "B2", "B3"])
    framing_levels: List[str] = field(default_factory=lambda: ["C1", "C2"])
    
    # Execution settings
    parallel: bool = True
    max_workers: int = 4
    device: str = "cuda"
    
    # API settings (for OpenRouter models)
    api_key_env: str = "OPENROUTER_API_KEY"
    api_base_url: str = "https://openrouter.ai/api/v1"
    api_reference_tokenizer_model_id: str = "gpt2"  # Reference tokenizer for API length matching
    
    # Ablation studies
    ablations: Dict[str, Any] = field(default_factory=lambda: {
        "partial_disagreement": False,
        "disagreement_rate": 0.5,
        "authority_but_uncertain": False,
        "aep_entity_virtualization": True,
        "aep_ratio": 0.5
    })
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: Path):
        """Save configuration to JSON file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        """Load configuration from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


def generate_trial_id(model_id: str, question_id: str, condition: str, replicate: int) -> str:
    """Generate unique trial ID"""
    timestamp = int(time.time() * 1000)
    raw = f"{model_id}_{question_id}_{condition}_{replicate}_{timestamp}"
    hash_id = hashlib.md5(raw.encode('utf-8')).hexdigest()[:12]
    return f"trial_{hash_id}"


def write_jsonl_record(record: TrialRecord, output_file: Path, lock=None):
    """
    Write a single trial record to JSONL file (thread-safe if lock provided).
    
    Args:
        record: TrialRecord to write
        output_file: Path to output JSONL file
        lock: Optional threading.Lock for thread-safe writes
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if lock:
        with lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(record.to_jsonl_line() + '\n')
    else:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(record.to_jsonl_line() + '\n')


def read_jsonl_records(input_file: Path) -> List[TrialRecord]:
    """
    Read trial records from JSONL file.
    
    Args:
        input_file: Path to input JSONL file
    
    Returns:
        List of TrialRecord objects
    """
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Reconstruct nested dataclasses
                pass1_data = data.pop('pass1')
                pass2_data = data.pop('pass2')
                
                pass1 = PassResult(**pass1_data)
                pass2 = PassResult(**pass2_data)
                
                record = TrialRecord(
                    **data,
                    pass1=pass1,
                    pass2=pass2
                )
                records.append(record)
    
    return records


def load_questions(questions_file: Path) -> List[Question]:
    """Load BEAT-300 questions from JSONL file"""
    questions = []
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Backward-compatibility: allow older field name "prompt"
                if "prompt_base" not in data and "prompt" in data:
                    data["prompt_base"] = data.pop("prompt")
                if data.get("input_sha256"):
                    digest = hashlib.sha256((data["prompt_base"] + "|" + data["excerpt"]).encode()).hexdigest()
                    if data["input_sha256"] != digest or data.get("hash") != digest[:16]:
                        raise ValueError(f"Input checksum mismatch: {data.get('question_id')}")
                    if data.get("excerpt_hash") != hashlib.sha256(data["excerpt"].encode()).hexdigest()[:16]:
                        raise ValueError(f"Excerpt checksum mismatch: {data.get('question_id')}")
                questions.append(Question(**data))
    return questions


def save_questions(questions: List[Question], output_file: Path):
    """Save questions to JSONL file"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for q in questions:
            f.write(json.dumps(q.to_dict(), ensure_ascii=False) + '\n')


def get_commit_hash() -> str:
    """Get current git commit hash (if in a git repo)"""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except:
        pass
    return "unknown"


# Example usage
if __name__ == "__main__":
    # Demonstrate data structures
    print("BEAT-300 Data Structures")
    print("=" * 60)
    
    # Example question
    q = Question(
        question_id="q001",
        domain="Medicine",
        prompt_base="Given the abstract excerpt below, does the evidence support answering YES to the following question?\n\nQuestion: Is aspirin effective for primary prevention of cardiovascular disease in adults over 60?\n\nAnswer with 'Yes' or 'No' on the first line, followed by 4-6 sentences explaining your reasoning based on the excerpt.",
        excerpt="Background: Aspirin use for primary prevention... Methods: Randomized trial of 10,000 patients... Conclusions: No significant reduction in cardiovascular events."
    )
    print("Question:")
    print(json.dumps(q.to_dict(), indent=2))
    print()
    
    # Example config
    config = ExperimentConfig()
    print("Default Configuration:")
    print(json.dumps(config.to_dict(), indent=2))
