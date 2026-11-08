#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Confidence-tier utilities for the benchmark.

Helper functions for stratifying items by their screening confidence.
"""

from typing import List, Dict, Any
from collections import Counter, defaultdict


def assign_confidence_tier(yes_proportion: float) -> str:
    """
    Assign confidence tier based on yes_proportion.
    
    Tier boundaries:
    - High: [0, 0.2] ∪ [0.8, 1.0] - Model is decisive
    - Medium: [0.2, 0.4] ∪ [0.6, 0.8] - Moderate uncertainty
    - Boundary: [0.4, 0.6] - True cognitive boundary
    
    Args:
        yes_proportion: Proportion of "Yes" responses (0.0 to 1.0)
    
    Returns:
        Confidence tier string: "High", "Medium", or "Boundary"
    """
    if 0.4 <= yes_proportion <= 0.6:
        return "Boundary"
    elif 0.2 <= yes_proportion <= 0.4 or 0.6 <= yes_proportion <= 0.8:
        return "Medium"
    else:  # [0, 0.2] or [0.8, 1.0]
        return "High"


def filter_by_tier(questions: List[Dict[str, Any]], tier: str) -> List[Dict[str, Any]]:
    """
    Filter questions by confidence tier.
    
    Args:
        questions: List of question dicts with 'confidence_tier' field
        tier: Target tier ("High", "Medium", or "Boundary")
    
    Returns:
        Filtered list of questions
    """
    return [q for q in questions if q.get('confidence_tier') == tier]


def filter_by_tiers(questions: List[Dict[str, Any]], tiers: List[str]) -> List[Dict[str, Any]]:
    """
    Filter questions by multiple confidence tiers.
    
    Args:
        questions: List of question dicts
        tiers: List of target tiers
    
    Returns:
        Filtered list of questions
    """
    tier_set = set(tiers)
    return [q for q in questions if q.get('confidence_tier') in tier_set]


def compute_tier_statistics(results: List[Dict[str, Any]]) -> Dict[str, Dict]:
    """
    Compute statistics stratified by confidence tier.
    
    Args:
        results: List of trial results with 'question_id' and 'reversal' fields
        questions: List of questions with 'question_id' and 'confidence_tier' fields
    
    Returns:
        Dict mapping tier to statistics
    """
    # Group results by tier
    tier_results = defaultdict(list)
    
    for result in results:
        tier = result.get('confidence_tier')
        if tier:
            tier_results[tier].append(result)
    
    # Compute statistics per tier
    tier_stats = {}
    
    for tier, tier_data in tier_results.items():
        reversals = [r for r in tier_data if r.get('reversal')]
        
        tier_stats[tier] = {
            'n_trials': len(tier_data),
            'n_reversals': len(reversals),
            'reversal_rate': len(reversals) / len(tier_data) if tier_data else 0.0,
            'tier': tier
        }
    
    return tier_stats


def get_tier_distribution(questions: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Get distribution of questions across confidence tiers.
    
    Args:
        questions: List of question dicts
    
    Returns:
        Dict mapping tier to count
    """
    return Counter(q.get('confidence_tier') for q in questions)


def stratify_by_domain_and_tier(questions: List[Dict[str, Any]]) -> Dict[str, Dict[str, List]]:
    """
    Stratify questions by domain and confidence tier.
    
    Args:
        questions: List of question dicts
    
    Returns:
        Nested dict: {domain: {tier: [questions]}}
    """
    stratified = defaultdict(lambda: defaultdict(list))
    
    for q in questions:
        domain = q.get('domain')
        tier = q.get('confidence_tier')
        if domain and tier:
            stratified[domain][tier].append(q)
    
    return dict(stratified)
