#!/usr/bin/env python3
"""
Re-extract chain-of-thought and final answers from existing model outputs.
Uses the improved extraction function without re-running inference.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Import the improved extraction function
from collect_data import extract_cot_and_answer, ModelOutput


def reextract_from_file(input_file: Path, output_file: Path) -> None:
    """
    Re-extract answers from existing model outputs JSON file.

    Args:
        input_file: Path to existing model_outputs.json
        output_file: Path to save fixed outputs
    """
    print(f"Loading existing outputs from: {input_file}")

    with open(input_file) as f:
        existing_data = json.load(f)

    print(f"Loaded {len(existing_data)} existing results")
    print("Re-extracting with improved function...")

    # Re-extract answers
    fixed_results = []
    issues_found = 0

    for i, entry in enumerate(existing_data):
        # Get the full response
        full_response = entry["full_response"]

        # Re-extract with improved function
        new_cot, new_answer = extract_cot_and_answer(full_response)

        # Compare with old extraction
        old_answer = entry["final_answer"]
        if old_answer != new_answer:
            issues_found += 1
            if issues_found <= 5:  # Show first 5 fixes
                print(f"\n[{i+1}] Problem: {entry['problem_id']}")
                print(f"  Old answer: '{old_answer[:50]}{'...' if len(old_answer) > 50 else ''}'")
                print(f"  New answer: '{new_answer[:50]}{'...' if len(new_answer) > 50 else ''}'")

        # Create updated entry
        entry["chain_of_thought"] = new_cot
        entry["final_answer"] = new_answer

        fixed_results.append(entry)

    print(f"\n✓ Fixed {issues_found} out of {len(existing_data)} extractions")

    # Save fixed results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(fixed_results, f, indent=2)

    print(f"✓ Saved fixed results to: {output_file}")

    # Validation check
    suspicious_count = 0
    for entry in fixed_results:
        answer = entry["final_answer"]
        if len(answer) < 2 or answer in ['**', '.', '', '...']:
            suspicious_count += 1
            if suspicious_count <= 3:
                print(f"  WARNING: Suspicious answer for {entry['problem_id']}: '{answer}'")

    if suspicious_count > 0:
        print(f"\n⚠ Found {suspicious_count} potentially suspicious extractions")
    else:
        print("\n✓ All extractions look valid!")


def main():
    # Default paths
    input_file = Path("data/raw/model_outputs.json")
    output_file = Path("data/raw/model_outputs_fixed.json")

    # Allow command line arguments
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2])

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    reextract_from_file(input_file, output_file)


if __name__ == "__main__":
    main()
