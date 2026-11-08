"""Experiment types and lazily loaded inference utilities.

Reading and validating input data requires only Python's standard library.
Inference-specific dependencies are loaded only when their features are used.
"""
from importlib import import_module

__version__ = "1.0.0"
__author__ = "Yupeng Tang and Mingfeng Lin"

from .data_structures import (
    Question, PassResult, TrialRecord, ExperimentConfig,
    load_questions, save_questions, read_jsonl_records, write_jsonl_record,
)

_LAZY_EXPORTS = {
    "yes_no_logodds": "mechanistic_probing",
    "classify_revision": "mechanistic_probing",
    "compute_drift_calibrated_threshold": "mechanistic_probing",
    "LogOddsResult": "mechanistic_probing",
    "RevisionTaxonomy": "mechanistic_probing",
    "TwoPassExperiment": "experiment_runner",
    "compute_length_matched_prompts": "prompt_utils",
    "verify_length_matching": "prompt_utils",
    "apply_aep_transformation": "aep_transformation",
    "should_apply_aep": "aep_transformation",
    "transform_question_to_aep": "aep_transformation",
    "AEPMetadata": "aep_transformation",
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("." + _LAZY_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "Question", "PassResult", "TrialRecord", "ExperimentConfig",
    "load_questions", "save_questions", "read_jsonl_records", "write_jsonl_record",
    *_LAZY_EXPORTS,
]
