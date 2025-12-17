#!/usr/bin/env python3
"""
Flip detection and annotation tool.
Automatically detects CoT/answer inconsistencies and categorizes flip types.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import Counter
import difflib

class FlipType(Enum):
    """Categories of last-minute answer flips."""
    
    # No flip - answer matches reasoning
    NO_FLIP = "no_flip"
    
    # Type 1: Numeric/comparison flips
    NUMERIC_REVERSAL = "numeric_reversal"  # CoT says A > B, answer says B
    CALCULATION_FLIP = "calculation_flip"  # CoT calculates X, answer says Y
    
    # Type 2: Logic flips
    LOGIC_REVERSAL = "logic_reversal"  # CoT concludes yes, answer says no
    NEGATION_FLIP = "negation_flip"  # Double negation confusion
    
    # Type 3: Format flips
    FORMAT_CONFUSION = "format_confusion"  # Right answer, wrong format
    UNIT_FLIP = "unit_flip"  # Forgot units or wrong units
    
    # Type 4: Hedging flips
    HEDGING_FLIP = "hedging_flip"  # CoT is confident, answer hedges
    CONFIDENCE_FLIP = "confidence_flip"  # CoT is uncertain, answer is confident
    
    # Type 5: Random/unclear
    APPARENT_RANDOM = "apparent_random"  # No clear pattern
    UNCLEAR = "unclear"  # Can't determine

@dataclass
class FlipAnnotation:
    """Detailed annotation of a potential flip."""
    problem_id: str
    model_name: str
    
    # Core detection
    is_flip: bool
    flip_type: str
    confidence: float  # 0-1, how confident in the annotation
    
    # Analysis
    cot_implied_answer: str
    final_answer: str
    correct_answer: str
    
    # Detailed breakdown
    cot_conclusion_sentence: str  # The sentence in CoT that implies the answer
    flip_location: Optional[int]  # Character position where flip occurs
    
    # Quality metrics
    cot_quality: str  # "good", "poor", "mixed"
    answer_matches_correct: bool
    cot_matches_correct: bool
    
    # Manual review fields
    manual_reviewed: bool = False
    manual_notes: str = ""
    reviewer_flip_type: Optional[str] = None

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.lower().strip()
    
    # Remove common prefixes/suffixes
    prefixes = ["the answer is", "answer:", "final answer:", "therefore", "thus", "so", "hence"]
    for prefix in prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
    
    # Remove punctuation
    answer = re.sub(r'[.,!?;:]$', '', answer)
    
    # Normalize whitespace
    answer = ' '.join(answer.split())
    
    # Handle common variations
    answer = answer.replace('$', '').replace('%', ' percent')
    
    # Handle yes/no variations
    yes_words = {'yes', 'true', 'correct', 'right', 'affirmative'}
    no_words = {'no', 'false', 'incorrect', 'wrong', 'negative'}
    
    if answer in yes_words:
        answer = 'yes'
    elif answer in no_words:
        answer = 'no'
    
    return answer

def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text."""
    # Match integers, decimals, and negative numbers
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    return [float(m) for m in matches if m not in ['-', '.']]

def find_conclusion_sentence(cot: str) -> Tuple[str, str]:
    """
    Find the sentence in CoT that states the conclusion.
    Returns (conclusion_sentence, implied_answer).
    """
    sentences = re.split(r'[.!?]\s+', cot)
    
    # Keywords that indicate conclusion
    conclusion_markers = [
        "therefore", "thus", "so", "hence", "this means",
        "the answer is", "we get", "we have", "this gives",
        "the result is", "which equals", "equals", "is equal to",
        "is larger", "is greater", "is smaller", "is less",
        "is true", "is false", "is correct", "is incorrect",
        "is yes", "is no", "must be", "should be"
    ]
    
    # Search from end (conclusions usually at end of reasoning)
    for sentence in reversed(sentences):
        sentence_lower = sentence.lower()
        for marker in conclusion_markers:
            if marker in sentence_lower:
                # Extract what comes after the marker
                idx = sentence_lower.find(marker)
                implied = sentence[idx + len(marker):].strip()
                implied = re.sub(r'^[:\s]+', '', implied)
                return sentence.strip(), implied
    
    # Fallback: return last substantive sentence
    for sentence in reversed(sentences):
        if len(sentence.strip()) > 10:
            return sentence.strip(), ""
    
    return "", ""

def detect_flip(
    question: str,
    cot: str,
    final_answer: str,
    correct_answer: str,
    problem_type: str = ""
) -> FlipAnnotation:
    """
    Detect if there's a flip between CoT and final answer.
    """
    # Normalize answers
    norm_final = normalize_answer(final_answer)
    norm_correct = normalize_answer(correct_answer)
    
    # Find what the CoT implies
    conclusion_sentence, cot_implied = find_conclusion_sentence(cot)
    norm_cot_implied = normalize_answer(cot_implied) if cot_implied else ""
    
    # Check if answers match
    answer_matches_correct = answers_match(norm_final, norm_correct)
    cot_matches_correct = answers_match(norm_cot_implied, norm_correct) if norm_cot_implied else None
    
    # Detect flip
    is_flip = False
    flip_type = FlipType.NO_FLIP
    confidence = 0.5
    
    # Strategy 1: Direct comparison of CoT implied vs final answer
    if norm_cot_implied and not answers_match(norm_cot_implied, norm_final):
        is_flip = True
        confidence = 0.8
        
        # Determine flip type
        flip_type = classify_flip_type(
            question, cot, final_answer, cot_implied,
            norm_final, norm_cot_implied, problem_type
        )
    
    # Strategy 2: Check for numeric contradictions
    cot_numbers = extract_numbers(conclusion_sentence)
    answer_numbers = extract_numbers(final_answer)
    
    if cot_numbers and answer_numbers:
        # Check if CoT computed one value but answer states another
        if cot_numbers[-1] != answer_numbers[0]:
            # This might be a flip
            if not is_flip:
                is_flip = True
                flip_type = FlipType.CALCULATION_FLIP
                confidence = 0.7
    
    # Strategy 3: Check for yes/no reversals
    cot_lower = cot.lower()
    final_lower = norm_final.lower()
    
    yes_in_cot = any(word in cot_lower for word in ['yes', 'true', 'correct', 'is larger', 'is greater'])
    no_in_cot = any(word in cot_lower for word in ['no', 'false', 'incorrect', 'is not', 'is smaller', 'is less'])
    
    if yes_in_cot and final_lower in ['no', 'false']:
        is_flip = True
        flip_type = FlipType.LOGIC_REVERSAL
        confidence = 0.85
    elif no_in_cot and final_lower in ['yes', 'true']:
        is_flip = True
        flip_type = FlipType.LOGIC_REVERSAL
        confidence = 0.85
    
    # Assess CoT quality
    cot_quality = assess_cot_quality(cot, question)
    
    return FlipAnnotation(
        problem_id="",  # Will be filled by caller
        model_name="",  # Will be filled by caller
        is_flip=is_flip,
        flip_type=flip_type.value,
        confidence=confidence,
        cot_implied_answer=cot_implied,
        final_answer=final_answer,
        correct_answer=correct_answer,
        cot_conclusion_sentence=conclusion_sentence,
        flip_location=find_flip_location(cot, final_answer) if is_flip else None,
        cot_quality=cot_quality,
        answer_matches_correct=answer_matches_correct,
        cot_matches_correct=cot_matches_correct if cot_matches_correct is not None else False,
    )

def answers_match(a: str, b: str, threshold: float = 0.8) -> bool:
    """Check if two answers match (with fuzzy matching)."""
    if a == b:
        return True
    
    # Try numeric comparison
    a_nums = extract_numbers(a)
    b_nums = extract_numbers(b)
    if a_nums and b_nums and abs(a_nums[0] - b_nums[0]) < 0.01:
        return True
    
    # Fuzzy string match
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return ratio >= threshold

def classify_flip_type(
    question: str, cot: str, final_answer: str, cot_implied: str,
    norm_final: str, norm_cot_implied: str, problem_type: str
) -> FlipType:
    """Classify the type of flip."""
    
    # Numeric problems
    if problem_type in ["decimal_comparison", "arithmetic_word_problem", "percentage"]:
        cot_nums = extract_numbers(cot_implied)
        final_nums = extract_numbers(final_answer)
        
        if cot_nums and final_nums and cot_nums != final_nums:
            if "comparison" in problem_type or "larger" in question.lower() or "greater" in question.lower():
                return FlipType.NUMERIC_REVERSAL
            return FlipType.CALCULATION_FLIP
    
    # Logic problems
    if problem_type == "logic_puzzle":
        if norm_cot_implied in ['yes', 'true'] and norm_final in ['no', 'false']:
            return FlipType.LOGIC_REVERSAL
        if norm_cot_implied in ['no', 'false'] and norm_final in ['yes', 'true']:
            return FlipType.LOGIC_REVERSAL
        if "not" in cot_implied.lower() or "not" in final_answer.lower():
            return FlipType.NEGATION_FLIP
    
    # Format issues
    if len(norm_final) < 5 and len(norm_cot_implied) > 20:
        return FlipType.FORMAT_CONFUSION
    
    # Hedging
    hedging_words = ['maybe', 'perhaps', 'possibly', 'might', 'could be', 'not sure']
    if any(w in final_answer.lower() for w in hedging_words):
        if not any(w in cot_implied.lower() for w in hedging_words):
            return FlipType.HEDGING_FLIP
    
    return FlipType.APPARENT_RANDOM

def assess_cot_quality(cot: str, question: str) -> str:
    """Assess the quality of the chain of thought."""
    
    # Length check
    if len(cot) < 50:
        return "poor"
    
    # Check for reasoning indicators
    reasoning_markers = [
        "because", "therefore", "thus", "since", "if", "then",
        "first", "second", "next", "finally", "step",
        "let me", "i need to", "we need to", "let's"
    ]
    
    cot_lower = cot.lower()
    marker_count = sum(1 for m in reasoning_markers if m in cot_lower)
    
    if marker_count >= 3:
        return "good"
    elif marker_count >= 1:
        return "mixed"
    else:
        return "poor"

def find_flip_location(cot: str, final_answer: str) -> int:
    """Find the approximate character position where the flip occurs."""
    # Look for transition to final answer
    markers = ["final answer", "answer:", "therefore", "thus", "so the answer"]
    
    cot_lower = cot.lower()
    for marker in markers:
        idx = cot_lower.rfind(marker)
        if idx != -1:
            return idx
    
    return len(cot) - len(final_answer)

def annotate_dataset(
    raw_data_path: Path,
    problems_path: Path,
    output_path: Path
) -> Dict[str, Any]:
    """
    Annotate a full dataset with flip detection.
    """
    # Load data
    with open(raw_data_path) as f:
        raw_data = json.load(f)
    
    with open(problems_path) as f:
        problems = {p["id"]: p for p in json.load(f)}
    
    annotations = []
    
    for item in raw_data:
        problem_id = item["problem_id"]
        problem = problems.get(problem_id, {})
        problem_type = problem.get("type", "")
        
        annotation = detect_flip(
            question=item["question"],
            cot=item["chain_of_thought"],
            final_answer=item["final_answer"],
            correct_answer=item["correct_answer"],
            problem_type=problem_type
        )
        
        annotation.problem_id = problem_id
        annotation.model_name = item["model_name"]
        
        # Add problem metadata
        annotation_dict = asdict(annotation)
        annotation_dict["problem_type"] = problem_type
        annotation_dict["problem_subtype"] = problem.get("subtype", "")
        annotation_dict["problem_flip_risk"] = problem.get("flip_risk", "")
        annotation_dict["full_response"] = item["full_response"]
        annotation_dict["chain_of_thought"] = item["chain_of_thought"]
        
        annotations.append(annotation_dict)
    
    # Save annotations
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    # Calculate statistics
    stats = calculate_stats(annotations)
    
    return stats

def calculate_stats(annotations: List[Dict]) -> Dict[str, Any]:
    """Calculate statistics from annotations."""
    
    total = len(annotations)
    flips = [a for a in annotations if a["is_flip"]]
    non_flips = [a for a in annotations if not a["is_flip"]]
    
    # By flip type
    flip_types = Counter(a["flip_type"] for a in flips)
    
    # By problem type
    flips_by_problem_type = Counter(a["problem_type"] for a in flips)
    total_by_problem_type = Counter(a["problem_type"] for a in annotations)
    
    flip_rate_by_type = {
        ptype: flips_by_problem_type.get(ptype, 0) / count
        for ptype, count in total_by_problem_type.items()
    }
    
    # By flip risk
    flips_by_risk = Counter(a["problem_flip_risk"] for a in flips)
    total_by_risk = Counter(a["problem_flip_risk"] for a in annotations)
    
    flip_rate_by_risk = {
        risk: flips_by_risk.get(risk, 0) / count if count > 0 else 0
        for risk, count in total_by_risk.items()
    }
    
    # Correctness analysis
    flip_correct = sum(1 for a in flips if a["answer_matches_correct"])
    flip_cot_correct = sum(1 for a in flips if a["cot_matches_correct"])
    nonflip_correct = sum(1 for a in non_flips if a["answer_matches_correct"])
    
    stats = {
        "total_examples": total,
        "total_flips": len(flips),
        "flip_rate": len(flips) / total if total > 0 else 0,
        
        "flip_types": dict(flip_types),
        
        "flip_rate_by_problem_type": flip_rate_by_type,
        "flip_rate_by_risk_level": flip_rate_by_risk,
        
        "accuracy": {
            "overall": sum(1 for a in annotations if a["answer_matches_correct"]) / total if total > 0 else 0,
            "flip_answer_correct": flip_correct / len(flips) if flips else 0,
            "flip_cot_would_be_correct": flip_cot_correct / len(flips) if flips else 0,
            "non_flip_correct": nonflip_correct / len(non_flips) if non_flips else 0,
        },
        
        "cot_quality": Counter(a["cot_quality"] for a in annotations),
        
        "high_confidence_flips": sum(1 for a in flips if a["confidence"] >= 0.8),
    }
    
    return stats

def print_stats(stats: Dict[str, Any]):
    """Pretty print statistics."""
    print("\n" + "="*60)
    print("FLIP DETECTION STATISTICS")
    print("="*60)
    
    print(f"\nTotal examples: {stats['total_examples']}")
    print(f"Total flips detected: {stats['total_flips']}")
    print(f"Overall flip rate: {stats['flip_rate']:.1%}")
    print(f"High confidence flips: {stats['high_confidence_flips']}")
    
    print("\n--- Flip Types ---")
    for ftype, count in sorted(stats['flip_types'].items(), key=lambda x: -x[1]):
        print(f"  {ftype}: {count}")
    
    print("\n--- Flip Rate by Problem Type ---")
    for ptype, rate in sorted(stats['flip_rate_by_problem_type'].items(), key=lambda x: -x[1]):
        print(f"  {ptype}: {rate:.1%}")
    
    print("\n--- Flip Rate by Risk Level ---")
    for risk, rate in sorted(stats['flip_rate_by_risk_level'].items(), key=lambda x: -x[1]):
        print(f"  {risk}: {rate:.1%}")
    
    print("\n--- Accuracy Analysis ---")
    acc = stats['accuracy']
    print(f"  Overall accuracy: {acc['overall']:.1%}")
    print(f"  Flip examples - answer correct: {acc['flip_answer_correct']:.1%}")
    print(f"  Flip examples - CoT would be correct: {acc['flip_cot_would_be_correct']:.1%}")
    print(f"  Non-flip accuracy: {acc['non_flip_correct']:.1%}")
    
    print("\n--- CoT Quality ---")
    for quality, count in stats['cot_quality'].items():
        print(f"  {quality}: {count}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Annotate flip dataset")
    parser.add_argument("--input", type=str, 
                       default="./data/raw/model_outputs.json")
    parser.add_argument("--problems", type=str,
                       default="./data/problems/all_problems.json")
    parser.add_argument("--output", type=str,
                       default="./data/annotated/flip_annotations.json")
    
    args = parser.parse_args()
    
    stats = annotate_dataset(
        Path(args.input),
        Path(args.problems),
        Path(args.output)
    )
    
    print_stats(stats)
    
    # Save stats
    stats_path = Path(args.output).parent / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_path}")

if __name__ == "__main__":
    main()
