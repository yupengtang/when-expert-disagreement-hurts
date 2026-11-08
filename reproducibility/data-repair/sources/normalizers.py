#!/usr/bin/env python3
"""
Candidate Pool Construction for BEAT-300

Build a pool of ~1-3K binary decision candidates from:
- PubMedQA: biomedical research questions with abstract excerpts
- SciFact: scientific claims with research abstracts (supporting/refuting)
- ContractNLI: contractual hypotheses with contract excerpt evidence

Per DESIGN.md §4.2: Candidate Pool and Unified Task Interface

Output: candidate_pool.jsonl with fields:
  candidate_id, domain, prompt_base, excerpt, ground_truth, source, hash, metadata
"""

import json
import hashlib
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Target candidate counts per source (initial pool ≈ 1-3K)
TARGET_PUBMEDQA = 1000
TARGET_SCIFACT = 300
TARGET_CONTRACTNLI = 800
TARGET_TOTAL = TARGET_PUBMEDQA + TARGET_SCIFACT + TARGET_CONTRACTNLI


def hash_text(text: str) -> str:
    """Generate hash for auditing."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


# ============================================================================
# PubMedQA: Medicine (Biomedical Research Questions)
# ============================================================================

def load_pubmedqa() -> List[Dict]:
    """
    Load PubMedQA dataset.
    
    Dataset structure:
    - question: research question (e.g., "Do preoperative statins reduce mortality?")
    - context: abstract sections (objectives, methods, results, conclusions)
    - long_answer: detailed reasoning
    - final_decision: "yes", "no", "maybe"
    
    We keep only {yes, no}, drop "maybe" per DESIGN.md §4.2.
    """
    try:
        from datasets import load_dataset
        print("Loading PubMedQA from HuggingFace...")
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        return list(ds)
    except Exception as e:
        print(f"Failed to load PubMedQA: {e}")
        print("Install: pip install datasets")
        return []


def normalize_pubmedqa_to_binary(item: Dict) -> Optional[Dict]:
    """
    Normalize PubMedQA to BEAT-300 format.
    
    Per DESIGN.md §3.2:
    - Task: "Given the abstract excerpt, does the evidence support answering Yes to the question?"
    - Binary: keep {yes, no}, drop maybe
    - Evidence: concatenate context sections into single excerpt
    - B2 evidence: use long_answer as strong relevant add-on
    """
    # Filter: only yes/no, drop maybe
    label = item.get("final_decision", "").lower()
    if label not in ["yes", "no"]:
        return None
    
    question = item.get("question", "").strip()
    if not question:
        return None
    
    # Extract abstract excerpt (concatenate context sections)
    contexts = item.get("context", {})
    if not contexts:
        return None
    
    # Concatenate abstract sections
    excerpt_parts = []
    for section_name, section_texts in contexts.items():
        if isinstance(section_texts, list):
            section_content = " ".join(section_texts)
        else:
            section_content = section_texts
        
        if section_content.strip():
            excerpt_parts.append(f"{section_name}: {section_content}")
    
    excerpt = "\n\n".join(excerpt_parts).strip()
    
    if not excerpt or len(excerpt) < 50:
        return None
    
    # Normalize to unified task interface (DESIGN.md §4.2)
    # prompt_base: binary only (for BEAT-300 screening — cheap)
    prompt_base = f"Given the abstract excerpt below, answer the following question:\n\nQuestion: {question}\n\nAnswer 'Yes' if the evidence supports this, or 'No' if it does not.\nAnswer ONLY with 'Yes' or 'No'. No explanation or additional text."
    # prompt_base_experiment: Yes/No + reasoning (for reversal-rate experiment)
    prompt_base_experiment = f"Given the abstract excerpt below, does the evidence support answering YES to the following question?\n\nQuestion: {question}\n\nAnswer with 'Yes' or 'No' on the first line, followed by 4-6 sentences explaining your reasoning based on the excerpt."
    
    # Extract B2 evidence (long answer / conclusion-style text)
    long_answer = item.get("long_answer", "").strip()
    
    return {
        "domain": "Medicine",
        "prompt_base": prompt_base,
        "prompt_base_experiment": prompt_base_experiment,
        "excerpt": excerpt,
        "ground_truth": label.capitalize(),  # "Yes" or "No"
        "source": "PubMedQA",
        "metadata": {
            "pubmed_id": item.get("pubid", ""),
            "original_question": question,
            "b2_evidence": long_answer if long_answer else None,
            "context_sections": list(contexts.keys())
        }
    }


def collect_pubmedqa_candidates(target_count: int = TARGET_PUBMEDQA) -> List[Dict]:
    """Collect PubMedQA candidates"""
    print(f"\n{'='*60}")
    print(f"Collecting PubMedQA Candidates (Target: {target_count})")
    print(f"{'='*60}")
    
    raw_data = load_pubmedqa()
    if not raw_data:
        print("Warning: PubMedQA data not available")
        return []
    
    print(f"Raw PubMedQA items: {len(raw_data)}")
    
    candidates = []
    for item in raw_data:
        normalized = normalize_pubmedqa_to_binary(item)
        if normalized:
            candidates.append(normalized)
        
        if len(candidates) >= target_count:
            break
    
    print(f"Collected {len(candidates)} PubMedQA candidates")
    
    # Label distribution
    label_counts = defaultdict(int)
    for c in candidates:
        label_counts[c["ground_truth"]] += 1
    print(f"  Label distribution: {dict(label_counts)}")
    
    return candidates


# ============================================================================
# SciFact: Science (Scientific Verification)
# ============================================================================

def load_scifact() -> Tuple[List[Dict], Dict]:
    """
    Load SciFact dataset.
    
    Dataset structure:
    - claims: scientific claims
    - corpus: research abstracts
    - claim-abstract pairs with labels: SUPPORTS, REFUTES, NOT_ENOUGH_INFO
    - evidence sentences (rationale): annotated sentences supporting/refuting
    
    We keep {SUPPORTS, REFUTES}, drop NOT_ENOUGH_INFO per DESIGN.md §4.2.
    If HuggingFace loading fails (dataset script blocked), falls back to the
    official SciFact tarball.
    """
    try:
        from datasets import load_dataset
        print("Loading SciFact from HuggingFace...")
        ds = load_dataset("copenlu/scifact", split="train")
        corpus_ds = load_dataset("copenlu/scifact_corpus", split="train")
        corpus = {str(item['doc_id']): item for item in corpus_ds}
        return list(ds), corpus
    except Exception as e:
        print(f"Failed to load SciFact via datasets: {e}")
        print("Falling back to official SciFact tarball...")
        return load_scifact_from_tarball()


def load_scifact_from_tarball() -> Tuple[List[Dict], Dict]:
    """Load SciFact from official tarball (claims + corpus)."""
    import tarfile
    import requests
    from pathlib import Path

    data_dir = Path("data/raw/scifact")
    tar_path = data_dir / "data.tar.gz"
    extract_dir = data_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if not tar_path.exists():
        url = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
        print(f"Downloading SciFact tarball: {url}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(tar_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print(f"Saved: {tar_path}")

    if not extract_dir.exists() or not (extract_dir / "claims_train.jsonl").exists():
        print(f"Extracting SciFact tarball to {data_dir}...")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(data_dir)

    claims = []
    for fname in ["claims_train.jsonl", "claims_dev.jsonl"]:
        fpath = extract_dir / fname
        if not fpath.exists():
            continue
        with fpath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                claims.append(json.loads(line))

    corpus = {}
    corpus_path = extract_dir / "corpus.jsonl"
    if corpus_path.exists():
        with corpus_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                doc_id = str(item.get("doc_id", "")).strip()
                if not doc_id:
                    continue
                abstract = item.get("abstract", [])
                if isinstance(abstract, list):
                    abstract_sentences = [s.strip() for s in abstract if isinstance(s, str) and s.strip()]
                    abstract_text = " ".join(abstract_sentences).strip()
                else:
                    abstract_text = str(abstract).strip()
                    abstract_sentences = [abstract_text] if abstract_text else []
                corpus[doc_id] = {
                    "title": item.get("title", ""),
                    "abstract": abstract_text,
                    "sentences": abstract_sentences
                }

    return claims, corpus


def normalize_scifact_to_binary(item: Dict, corpus: Dict) -> Optional[Dict]:
    """
    Normalize SciFact to BEAT-300 format.
    
    Per DESIGN.md §4.2:
    - Task: "Given the abstract, does the evidence SUPPORT the claim?"
    - Binary: keep {SUPPORTS, REFUTES}, drop NOT_ENOUGH_INFO
    - Evidence: research abstract from corpus
    - B2 evidence: annotated rationale sentences
    """
    claim = item.get("claim", "").strip()
    if not claim:
        return None
    
    # Get cited docs with evidence
    cited_docs = item.get("cited_doc_ids", [])
    if not cited_docs:
        return None
    
    # Use first cited doc with evidence
    doc_id = str(cited_docs[0])
    if doc_id not in corpus:
        return None
    
    abstract_text = corpus[doc_id].get("abstract", "").strip()
    if not abstract_text or len(abstract_text) < 50:
        return None
    
    # Get label: SUPPORTS or REFUTES
    # SciFact labels are per doc-claim pair
    evidence_list = item.get("evidence", {})
    if not evidence_list or str(doc_id) not in evidence_list:
        return None

    doc_evidence = evidence_list[str(doc_id)]

    # SciFact tarball uses a list of evidence dicts per doc_id
    label = ""
    rationale_sentences = []
    if isinstance(doc_evidence, list):
        for ev in doc_evidence:
            ev_label = str(ev.get("label", "")).upper()
            if ev_label in ["SUPPORT", "SUPPORTS", "REFUTE", "REFUTES"]:
                label = "SUPPORTS" if ev_label.startswith("SUPPORT") else "REFUTES"
                rationale_sentences.extend(ev.get("sentences", []))
    elif isinstance(doc_evidence, dict):
        label = str(doc_evidence.get("label", "")).upper()
        rationale_sentences = doc_evidence.get("sentences", [])

    if label not in ["SUPPORTS", "REFUTES"]:
        return None
    
    # Get sentence texts from abstract
    abstract_sentences = corpus[doc_id].get("sentences", [])
    b2_rationale_texts = []
    for sent_id in rationale_sentences:
        if sent_id < len(abstract_sentences):
            b2_rationale_texts.append(abstract_sentences[sent_id])
    
    b2_evidence = " ".join(b2_rationale_texts) if b2_rationale_texts else None
    
    # Normalize to unified task interface
    # prompt_base: binary only (for BEAT-300 screening)
    prompt_base = f"Given the abstract below, evaluate the following claim:\n\nClaim: {claim}\n\nAnswer 'Yes' if the evidence supports the claim, or 'No' if it refutes the claim.\nAnswer ONLY with 'Yes' or 'No'. No explanation or additional text."
    # prompt_base_experiment: Yes/No + reasoning (for reversal experiment)
    prompt_base_experiment = f"Given the abstract below, does the evidence SUPPORT the following claim?\n\nClaim: {claim}\n\nAnswer with 'Yes' or 'No' on the first line, followed by 4-6 sentences explaining your reasoning based on the abstract."
    
    # Map label: SUPPORTS → Yes, REFUTES → No
    ground_truth = "Yes" if label == "SUPPORTS" else "No"
    
    return {
        "domain": "Science",
        "prompt_base": prompt_base,
        "prompt_base_experiment": prompt_base_experiment,
        "excerpt": abstract_text,
        "ground_truth": ground_truth,
        "source": "SciFact",
        "metadata": {
            "doc_id": doc_id,
            "original_claim": claim,
            "original_label": label,
            "b2_evidence": b2_evidence,
            "rationale_sentence_ids": rationale_sentences
        }
    }


def collect_scifact_candidates(target_count: int = TARGET_SCIFACT) -> List[Dict]:
    """Collect SciFact candidates"""
    print(f"\n{'='*60}")
    print(f"Collecting SciFact Candidates (Target: {target_count})")
    print(f"{'='*60}")
    
    raw_data, corpus = load_scifact()
    if not raw_data or not corpus:
        print("Warning: SciFact data not available")
        return []
    
    print(f"Raw SciFact claims: {len(raw_data)}")
    print(f"Abstract corpus: {len(corpus)} documents")
    
    candidates = []
    for item in raw_data:
        normalized = normalize_scifact_to_binary(item, corpus)
        if normalized:
            candidates.append(normalized)
        
        if len(candidates) >= target_count:
            break
    
    print(f"Collected {len(candidates)} SciFact candidates")
    
    # Label distribution
    label_counts = defaultdict(int)
    for c in candidates:
        label_counts[c["ground_truth"]] += 1
    print(f"  Label distribution: {dict(label_counts)}")
    
    return candidates


# ============================================================================
# ContractNLI: Law (Contractual Inference)
# ============================================================================

def load_contractnli() -> List[Dict]:
    """
    Load ContractNLI dataset.
    
    Dataset structure:
    - premise: contract excerpt
    - hypothesis: statement about contract
    - label: Entailment, Contradiction, NotMentioned
    - evidence: span annotations (start/end positions in premise)
    
    We keep {Entailment, Contradiction}, drop NotMentioned per DESIGN.md §4.2.
    """
    try:
        from datasets import load_dataset
        print("Loading ContractNLI from HuggingFace...")
        ds = load_dataset("coastalcph/contract_nli", split="train")
        return list(ds)
    except Exception as e:
        print(f"Failed to load ContractNLI: {e}")
        print("Trying alternative: kiddothe2b/contract-nli...")
        try:
            ds = load_dataset("kiddothe2b/contract-nli", split="train")
            return list(ds)
        except Exception as e2:
            print(f"Failed: {e2}")
            print("Install: pip install datasets")
            return []


def normalize_contractnli_to_binary(item: Dict) -> Optional[Dict]:
    """
    Normalize ContractNLI to BEAT-300 format.
    
    Per DESIGN.md §4.2:
    - Task: "Given the contract excerpt, is the hypothesis ENTAILED?"
    - Binary: keep {Entailment, Contradiction}, drop NotMentioned
    - Evidence: contract premise
    - B2 evidence: annotated evidence spans
    """
    premise = item.get("premise", "").strip()
    hypothesis = item.get("hypothesis", "").strip()
    
    if not premise or not hypothesis:
        return None
    
    if len(premise) < 50:
        return None
    
    # Get label
    label = item.get("label", "")
    if isinstance(label, int):
        # Map numeric labels: 0=Entailment, 1=Contradiction, 2=NotMentioned
        label_map = {0: "Entailment", 1: "Contradiction", 2: "NotMentioned"}
        label = label_map.get(label, "")
    
    label_str = str(label).strip()
    
    if label_str not in ["Entailment", "Contradiction"]:
        return None
    
    # Extract evidence spans (B2 evidence)
    evidence_spans = item.get("evidence", [])
    b2_evidence_text = None
    
    if evidence_spans:
        # Extract spans from premise
        span_texts = []
        for span in evidence_spans:
            start = span.get("start", 0)
            end = span.get("end", 0)
            if start < end and end <= len(premise):
                span_texts.append(premise[start:end])
        
        if span_texts:
            b2_evidence_text = " ".join(span_texts)
    
    # Normalize to unified task interface
    # prompt_base: binary only (for BEAT-300 screening)
    prompt_base = f"Given the contract excerpt below, evaluate the following hypothesis:\n\nHypothesis: {hypothesis}\n\nAnswer 'Yes' if the hypothesis is entailed by the contract, or 'No' if it is contradicted.\nAnswer ONLY with 'Yes' or 'No'. No explanation or additional text."
    # prompt_base_experiment: Yes/No + reasoning (for reversal experiment)
    prompt_base_experiment = f"Given the contract excerpt below, is the following hypothesis ENTAILED by the contract?\n\nHypothesis: {hypothesis}\n\nAnswer with 'Yes' or 'No' on the first line, followed by 4-6 sentences explaining your reasoning based on the contract excerpt."
    
    # Map label: Entailment → Yes, Contradiction → No
    ground_truth = "Yes" if label_str == "Entailment" else "No"
    
    return {
        "domain": "Law",
        "prompt_base": prompt_base,
        "prompt_base_experiment": prompt_base_experiment,
        "excerpt": premise,
        "ground_truth": ground_truth,
        "source": "ContractNLI",
        "metadata": {
            "original_hypothesis": hypothesis,
            "original_label": label_str,
            "b2_evidence": b2_evidence_text,
            "evidence_span_count": len(evidence_spans) if evidence_spans else 0
        }
    }


def collect_contractnli_candidates(target_count: int = TARGET_CONTRACTNLI) -> List[Dict]:
    """Collect ContractNLI candidates"""
    print(f"\n{'='*60}")
    print(f"Collecting ContractNLI Candidates (Target: {target_count})")
    print(f"{'='*60}")
    
    raw_data = load_contractnli()
    if not raw_data:
        print("Warning: ContractNLI data not available")
        return []
    
    print(f"Raw ContractNLI items: {len(raw_data)}")
    
    candidates = []
    for item in raw_data:
        normalized = normalize_contractnli_to_binary(item)
        if normalized:
            candidates.append(normalized)
        
        if len(candidates) >= target_count:
            break
    
    print(f"Collected {len(candidates)} ContractNLI candidates")
    
    # Label distribution
    label_counts = defaultdict(int)
    for c in candidates:
        label_counts[c["ground_truth"]] += 1
    print(f"  Label distribution: {dict(label_counts)}")
    
    return candidates


# ============================================================================
# Main Pipeline
# ============================================================================

def build_candidate_pool(
    output_file: Path,
    target_pubmedqa: int = TARGET_PUBMEDQA,
    target_scifact: int = TARGET_SCIFACT,
    target_contractnli: int = TARGET_CONTRACTNLI
) -> List[Dict]:
    """
    Build complete candidate pool from three sources.
    
    Returns list of candidates with structure:
    {
        "candidate_id": "cand_XXXX",
        "domain": "Medicine" | "Science" | "Law",
        "prompt_base": "...",
        "excerpt": "...",
        "ground_truth": "Yes" | "No",
        "source": "PubMedQA" | "SciFact" | "ContractNLI",
        "hash": "...",
        "metadata": {...}
    }
    """
    print("\n" + "="*60)
    print("BEAT-300 Candidate Pool Construction")
    print("="*60)
    print(f"Target: {target_pubmedqa + target_scifact + target_contractnli} candidates")
    print(f"  - PubMedQA (Medicine): {target_pubmedqa}")
    print(f"  - SciFact (Science): {target_scifact}")
    print(f"  - ContractNLI (Law): {target_contractnli}")
    print("="*60)
    
    # Collect from each source
    candidates = []
    
    # 1. PubMedQA
    pubmedqa_cands = collect_pubmedqa_candidates(target_pubmedqa)
    candidates.extend(pubmedqa_cands)
    
    # 2. SciFact
    scifact_cands = collect_scifact_candidates(target_scifact)
    candidates.extend(scifact_cands)
    
    # 3. ContractNLI
    contractnli_cands = collect_contractnli_candidates(target_contractnli)
    candidates.extend(contractnli_cands)
    
    # Assign candidate IDs and compute hashes
    for i, cand in enumerate(candidates):
        cand["candidate_id"] = f"cand_{i+1:04d}"
        combined = f"{cand['prompt_base']}|{cand['excerpt']}"
        cand["hash"] = hash_text(combined)
    
    # Summary statistics
    print("\n" + "="*60)
    print("Candidate Pool Summary")
    print("="*60)
    print(f"Total candidates: {len(candidates)}")
    
    domain_counts = defaultdict(int)
    source_counts = defaultdict(int)
    label_counts = defaultdict(int)
    
    for cand in candidates:
        domain_counts[cand["domain"]] += 1
        source_counts[cand["source"]] += 1
        label_counts[cand["ground_truth"]] += 1
    
    print("\nDomain distribution:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count} ({count/len(candidates)*100:.1f}%)")
    
    print("\nSource distribution:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count} ({count/len(candidates)*100:.1f}%)")
    
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({count/len(candidates)*100:.1f}%)")
    
    # Save to JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for cand in candidates:
            f.write(json.dumps(cand, ensure_ascii=False) + '\n')
    
    print(f"\nCandidate pool saved to: {output_file}")
    print("="*60)
    
    return candidates


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build BEAT-300 candidate pool")
    parser.add_argument("--output", type=str, default="data/candidate_pool.jsonl",
                       help="Output JSONL file")
    parser.add_argument("--pubmedqa", type=int, default=TARGET_PUBMEDQA,
                       help="Target count for PubMedQA")
    parser.add_argument("--scifact", type=int, default=TARGET_SCIFACT,
                       help="Target count for SciFact")
    parser.add_argument("--contractnli", type=int, default=TARGET_CONTRACTNLI,
                       help="Target count for ContractNLI")
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    candidates = build_candidate_pool(
        output_path,
        target_pubmedqa=args.pubmedqa,
        target_scifact=args.scifact,
        target_contractnli=args.contractnli
    )
    
    print(f"\nDone. Collected {len(candidates)} candidates.")
    print(f"Next step: Run boundary selection to generate BEAT-300")
