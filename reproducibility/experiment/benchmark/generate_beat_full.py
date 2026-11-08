#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble the benchmark from the candidate pool and the screening log.

Merges candidate_pool.jsonl with the screening results in selection_log.jsonl
and writes every screened question with its confidence tier, rather than only
the boundary cases.
"""

import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional
import argparse


def assign_confidence_tier(yes_proportion: float) -> str:
    """
    Assign confidence tier based on yes_proportion.
    
    Tier boundaries:
    - High: [0, 0.2] ∪ [0.8, 1.0] - Model is decisive
    - Medium: [0.2, 0.4] ∪ [0.6, 0.8] - Moderate uncertainty
    - Boundary: [0.4, 0.6] - True cognitive boundary
    """
    if 0.4 <= yes_proportion <= 0.6:
        return "Boundary"
    elif 0.2 <= yes_proportion <= 0.4 or 0.6 <= yes_proportion <= 0.8:
        return "Medium"
    else:  # [0, 0.2] or [0.8, 1.0]
        return "High"


def load_profiling_results(profiling_log_path: Path) -> Dict[str, Dict]:
    """
    Load profiling results from selection_log.jsonl.
    
    Returns:
        Dict mapping excerpt_hash to profiling data with yes_proportions
    """
    profiling_data = {}
    
    with open(profiling_log_path) as f:
        for line in f:
            if not line.strip():
                continue
            
            record = json.loads(line)
            
            if record.get('record_type') == 'candidate':
                # Use excerpt_hash as key for matching
                excerpt_hash = record.get('excerpt_hash')
                
                if not excerpt_hash:
                    continue
                
                # Extract yes_proportion from each screening model
                screening_results = record.get('screening_results', [])
                
                profiling_data[excerpt_hash] = {
                    'candidate_id': record.get('candidate_id'),
                    'domain': record.get('domain'),
                    'source': record.get('source'),
                    'screening_results': screening_results,
                    'retained': record.get('retained', False),
                    'excerpt_hash': excerpt_hash
                }
    
    return profiling_data


def generate_beat_full_benchmark(
    candidates_path: Path,
    profiling_log_path: Path,
    output_path: Path,
    min_valid_ratio: float = 0.8
) -> None:
    """
    Generate BEAT-FULL benchmark with confidence tier assignments.
    
    Args:
        candidates_path: Path to candidate_pool.jsonl
        profiling_log_path: Path to selection_log.jsonl (profiling results)
        output_path: Path to output beat_full_questions.jsonl
        min_valid_ratio: Minimum valid response ratio to include question
    """
    print(f"Loading profiling results from {profiling_log_path}...")
    profiling_data = load_profiling_results(profiling_log_path)
    print(f"  Loaded profiling for {len(profiling_data)} candidates")
    
    print(f"\nLoading candidates from {candidates_path}...")
    candidates = []
    with open(candidates_path) as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    print(f"  Loaded {len(candidates)} candidates")
    
    # Generate BEAT-FULL questions
    beat_full_questions = []
    tier_counts = Counter()
    domain_counts = Counter()
    excluded_count = 0
    excluded_reasons = Counter()
    
    for idx, candidate in enumerate(candidates, 1):
        candidate_id = candidate.get('candidate_id', f'cand_{idx:05d}')
        
        # Compute excerpt hash for matching with profiling data
        excerpt = candidate.get('excerpt', '')
        excerpt_hash = hashlib.sha256(excerpt.encode()).hexdigest()[:16]
        
        # Get profiling results using excerpt_hash
        profiling = profiling_data.get(excerpt_hash)
        
        if not profiling:
            excluded_count += 1
            excluded_reasons['no_profiling_data'] += 1
            continue
        
        screening_results = profiling.get('screening_results', [])
        
        if not screening_results:
            excluded_count += 1
            excluded_reasons['no_screening_results'] += 1
            continue
        
        # Check quality: at least one model should have high valid_ratio
        valid_results = [r for r in screening_results if r.get('valid_ratio', 0) >= min_valid_ratio]
        
        if not valid_results:
            excluded_count += 1
            excluded_reasons['low_valid_ratio'] += 1
            continue
        
        # Use first valid result for tier assignment (typically Llama-8B)
        primary_result = valid_results[0]
        yes_proportion = primary_result.get('yes_proportion', 0.5)
        
        # Assign confidence tier
        confidence_tier = assign_confidence_tier(yes_proportion)
        
        # Build BEAT-FULL question entry
        question = {
            'question_id': f'q{idx:04d}',
            'candidate_id': candidate_id,
            'domain': candidate.get('domain'),
            'source': candidate.get('source'),
            'prompt_base': candidate.get('prompt_base_experiment') or candidate.get('prompt_base'),
            'excerpt': candidate.get('excerpt'),
            'ground_truth': candidate.get('ground_truth'),
            'confidence_tier': confidence_tier,
            'yes_proportion_llama': None,
            'yes_proportion_mistral': None,
            'metadata': candidate.get('metadata', {}),
            'hash': candidate.get('hash', hashlib.sha256(candidate.get('excerpt', '').encode()).hexdigest()[:16])
        }
        
        # Extract yes_proportions by model
        for result in screening_results:
            model_id = result.get('model_id', '')
            yes_prop = result.get('yes_proportion')
            
            if 'Llama' in model_id or 'llama' in model_id:
                question['yes_proportion_llama'] = yes_prop
            elif 'Mistral' in model_id or 'mistral' in model_id:
                question['yes_proportion_mistral'] = yes_prop
        
        beat_full_questions.append(question)
        tier_counts[confidence_tier] += 1
        domain_counts[candidate.get('domain')] += 1
    
    # Write output
    print(f"\nWriting BEAT-FULL benchmark to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for question in beat_full_questions:
            f.write(json.dumps(question) + '\n')
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"BEAT-FULL Benchmark Generation Complete")
    print(f"{'='*60}")
    print(f"\nTotal questions: {len(beat_full_questions)}")
    print(f"Excluded: {excluded_count}")
    
    if excluded_reasons:
        print(f"\nExclusion reasons:")
        for reason, count in excluded_reasons.most_common():
            print(f"  {reason}: {count}")
    
    print(f"\nConfidence tier distribution:")
    for tier in ['High', 'Medium', 'Boundary']:
        count = tier_counts[tier]
        pct = 100 * count / len(beat_full_questions) if beat_full_questions else 0
        print(f"  {tier:12s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nDomain distribution:")
    for domain, count in domain_counts.most_common():
        pct = 100 * count / len(beat_full_questions) if beat_full_questions else 0
        print(f"  {domain:12s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nOutput written to: {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate BEAT-FULL benchmark from profiling data'
    )
    parser.add_argument(
        '--candidates',
        type=Path,
        default=Path('../runs/candidate_pool.jsonl'),
        help='Path to candidate_pool.jsonl'
    )
    parser.add_argument(
        '--profiling-log',
        type=Path,
        default=Path('../runs/selection_log.jsonl'),
        help='Path to profiling log (selection_log.jsonl)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('../runs/benchmark_questions.jsonl'),
        help='Output path for BEAT-FULL benchmark'
    )
    parser.add_argument(
        '--min-valid-ratio',
        type=float,
        default=0.8,
        help='Minimum valid response ratio (default: 0.8)'
    )
    
    args = parser.parse_args()
    
    generate_beat_full_benchmark(
        candidates_path=args.candidates,
        profiling_log_path=args.profiling_log,
        output_path=args.output,
        min_valid_ratio=args.min_valid_ratio
    )


if __name__ == '__main__':
    main()
