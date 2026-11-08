#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEP (Anonymized Entity Protocol) Transformation

Replaces real-world authority-bearing entities with fictional placeholders 
to test whether social susceptibility depends on real-world priors.

Optional ablation, disabled in every reported run (configs/main_audit.json):
- Medicine/Science: virtualize journals, institutions, guideline names, lab names
  (preserve biomedical entities and statistical expressions)
- Law: virtualize courts/companies/parties
  (preserve contractual obligations, quantifiers, and logical structure)

This decouples "authority identity" from "evidence content" without corrupting 
domain semantics.
"""

import re
from typing import Dict, Tuple
from dataclasses import dataclass


# Entity mapping: real to fictional, domain-aware

# Medicine/Science: Journals, Institutions, Guidelines
MEDICAL_SCIENCE_MAPPINGS = {
    # Medical Journals
    "New England Journal of Medicine": "Journal of Advanced Medicine",
    "NEJM": "Journal of Advanced Medicine",
    "The Lancet": "Global Medical Review",
    "JAMA": "Medical Science Quarterly",
    "Journal of the American Medical Association": "Medical Science Quarterly",
    "BMJ": "Clinical Research Journal",
    "British Medical Journal": "Clinical Research Journal",
    "Nature Medicine": "Biomedical Discoveries",
    "Cell": "Molecular Biology Reports",
    
    # Medical Institutions
    "Mayo Clinic": "Advanced Medical Center",
    "Johns Hopkins": "University Medical Institute",
    "Harvard Medical School": "National Medical School",
    "Cleveland Clinic": "Regional Health Institute",
    "Massachusetts General Hospital": "Metropolitan General Hospital",
    
    # Guidelines & Organizations
    "CDC": "National Health Agency",
    "Centers for Disease Control": "National Health Agency",
    "WHO": "Global Health Organization",
    "World Health Organization": "Global Health Organization",
    "FDA": "Federal Regulatory Agency",
    "Food and Drug Administration": "Federal Regulatory Agency",
    "USPSTF": "National Prevention Task Force",
    "American Heart Association": "Cardiovascular Research Society",
    "AHA": "Cardiovascular Research Society",
    "American College of Cardiology": "National Cardiology Institute",
    "ACC": "National Cardiology Institute",
    
    # Research Labs
    "NIH": "National Research Institute",
    "National Institutes of Health": "National Research Institute",
    "Cold Spring Harbor Laboratory": "Molecular Research Laboratory",
    "Broad Institute": "Genomics Research Center",
}

# Law: Courts, Companies, Legal Entities
LAW_MAPPINGS = {
    # Courts
    "Supreme Court": "High Court",
    "U.S. Supreme Court": "Federal High Court",
    "Court of Appeals": "Appellate Tribunal",
    "District Court": "Regional Court",
    "Ninth Circuit": "Regional Circuit",
    "Second Circuit": "Metropolitan Circuit",
    
    # Generic Company Names (preserve structure, virtualize identity)
    "Google": "TechCorp Alpha",
    "Microsoft": "Software Solutions Inc",
    "Amazon": "Commerce Platform Ltd",
    "Apple": "Technology Devices Co",
    "Meta": "Social Networks Inc",
    "Facebook": "Social Networks Inc",
}

# Combine all mappings
ENTITY_MAPPINGS = {
    **MEDICAL_SCIENCE_MAPPINGS,
    **LAW_MAPPINGS,
    "NATO": "Alliance N-6",
    
    # Universities
    "Harvard": "Apex University",
    "Harvard University": "Apex University",
    "MIT": "Institute T-1",
    "Massachusetts Institute of Technology": "Institute T-1",
    "Stanford": "Summit University",
    "Stanford University": "Summit University",
    "Oxford": "Prime Academy",
    "University of Oxford": "Prime Academy",
    "Cambridge": "Crown University",
    "University of Cambridge": "Crown University",
    
    # Companies
    "Google": "TechCorp Alpha",
    "Microsoft": "SoftSys Beta",
    "Apple": "DeviceInc Gamma",
    "Amazon": "MarketHub Delta",
    "Facebook": "SocialNet Epsilon",
    "Meta": "SocialNet Epsilon",
    "Tesla": "ElectriCar Zeta",
    "SpaceX": "RocketCo Eta",
    
    # Cities
    "New York": "Metropolis Prime",
    "New York City": "Metropolis Prime",
    "London": "Capital City Alpha",
    "Paris": "Central City Beta",
    "Tokyo": "Metro Gamma",
    "Beijing": "Capital City Delta",
    "Washington": "Federal City",
    "Washington D.C.": "Federal City",
    "Washington DC": "Federal City",
    
    # Historical figures (for legal/historical questions)
    "Supreme Court": "High Tribunal",
    "Congress": "Legislative Assembly",
    "Senate": "Upper Chamber",
    "House of Representatives": "Lower Chamber",
    "Constitution": "Foundational Charter",
    "First Amendment": "Charter Article 1",
    "Second Amendment": "Charter Article 2",
}


def apply_aep_transformation(text: str, entity_map: Dict[str, str] = None) -> Tuple[str, Dict[str, str]]:
    """
    Apply AEP transformation to replace real entities with fictional ones.
    
    Args:
        text: Original text with real entities
        entity_map: Custom entity mapping (if None, uses default)
    
    Returns:
        Tuple of (transformed_text, entities_replaced)
    """
    if entity_map is None:
        entity_map = ENTITY_MAPPINGS
    
    transformed = text
    replaced = {}
    
    # Sort by length (descending) to handle "United States" before "States"
    sorted_entities = sorted(entity_map.items(), key=lambda x: len(x[0]), reverse=True)
    
    for real_entity, fictional_entity in sorted_entities:
        # Case-insensitive replacement with word boundaries
        pattern = r'\b' + re.escape(real_entity) + r'\b'
        
        if re.search(pattern, transformed, re.IGNORECASE):
            transformed = re.sub(pattern, fictional_entity, transformed, flags=re.IGNORECASE)
            replaced[real_entity] = fictional_entity
    
    return transformed, replaced


def should_apply_aep(question_id: str, aep_ratio: float = 0.5) -> bool:
    """
    Determine if AEP should be applied to a question.
    
    Uses deterministic hashing to ensure consistent 50/50 split.
    
    Args:
        question_id: Question identifier
        aep_ratio: Proportion of questions to apply AEP (default: 0.5)
    
    Returns:
        True if AEP should be applied
    """
    # Deterministic assignment across runs (avoid Python's randomized hash()).
    # Use SHA256(question_id) to map into [0, 1).
    import hashlib
    h = hashlib.sha256(question_id.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / float(0xFFFFFFFF)
    return bucket < float(aep_ratio)


def transform_question_to_aep(question_text: str) -> Tuple[str, bool, Dict[str, str]]:
    """
    Transform a question to AEP version if it contains real-world entities.
    
    Args:
        question_text: Original question text
    
    Returns:
        Tuple of (transformed_text, was_transformed, entities_replaced)
    """
    transformed, replaced = apply_aep_transformation(question_text)
    
    # Only mark as AEP if entities were actually replaced
    was_transformed = len(replaced) > 0
    
    return transformed, was_transformed, replaced


@dataclass
class AEPMetadata:
    """Metadata for AEP transformation"""
    is_aep: bool
    entities_replaced: Dict[str, str]
    original_text: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "is_aep": self.is_aep,
            "entities_replaced": self.entities_replaced,
            "n_entities_replaced": len(self.entities_replaced)
        }


# Example usage
if __name__ == "__main__":
    print("AEP Transformation Module")
    print("=" * 60)
    
    # Test examples
    examples = [
        "Does the First Amendment protect commercial speech in the United States?",
        "Should the CDC recommend aspirin for cardiovascular disease?",
        "Is Harvard University located in Cambridge, Massachusetts?",
        "Does the European Union have a common currency?"
    ]
    
    print("\nTest Transformations:")
    print("-" * 60)
    
    for i, text in enumerate(examples, 1):
        transformed, replaced = apply_aep_transformation(text)
        print(f"\n{i}. Original:")
        print(f"   {text}")
        print(f"   Transformed:")
        print(f"   {transformed}")
        print(f"   Entities replaced: {len(replaced)}")
        if replaced:
            for real, fictional in replaced.items():
                print(f"     • {real} → {fictional}")
    
    print("\n" + "=" * 60)
    print("AEP ensures 50% real-world, 50% virtualized entities")
    print("for testing social susceptibility dependence on priors.")

