#!/usr/bin/env python3
"""
Package the flip detection dataset for public release.
Creates multiple formats and documentation.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
import shutil

def create_readme(output_dir: Path, stats: dict):
    """Create a comprehensive README for the dataset."""
    
    readme = f"""# CoT Flip Detection Dataset

## Overview

This dataset contains {stats['summary']['total_examples']} examples of reasoning model outputs 
annotated for "last-minute answer flips" - cases where the model's chain of thought 
implies one answer but the final output contradicts it.

**Created:** {datetime.now().strftime('%Y-%m-%d')}
**Author:** Syed Amaan
**License:** MIT

## What are Last-Minute Answer Flips?

A "flip" occurs when a reasoning model:
1. Produces a coherent chain of thought (CoT) that implies answer A
2. But outputs answer B as the final response

Example:
```
Question: Which is larger, 9.11 or 9.8?

CoT: Let me compare these decimals. 9.11 = 9.110 and 9.8 = 9.800.
     Comparing: 9.800 > 9.110, so 9.8 is larger.

Final Answer: 9.11 is larger.
```

## Dataset Statistics

- **Total examples:** {stats['summary']['total_examples']}
- **Flips detected:** {stats['summary']['total_flips']} ({stats['summary']['flip_rate']:.1%})
- **High-confidence flips:** {stats['summary']['high_confidence_flips']}

### Flip Types
"""
    for ftype, count in sorted(stats.get('by_flip_type', {}).items(), key=lambda x: -x[1]):
        readme += f"- `{ftype}`: {count}\n"
    
    readme += """
### Problem Types Included
"""
    for ptype, data in stats.get('by_problem_type', {}).items():
        readme += f"- `{ptype}`: {data['total']} examples, {data['flip_rate']:.1%} flip rate\n"
    
    readme += """
## Files

```
dataset/
├── README.md                    # This file
├── flip_dataset.json           # Full dataset in JSON format
├── flip_dataset.csv            # Dataset in CSV format (flattened)
├── problems/
│   └── all_problems.json       # Original problem set
├── annotations/
│   └── flip_annotations.json   # Detailed flip annotations
├── examples/
│   └── flip_examples.json      # 20 representative flip examples
└── analysis/
    ├── detailed_stats.json     # Comprehensive statistics
    ├── analysis_report.txt     # Human-readable report
    └── figures/                # Visualization PNGs
```

## Data Schema

### flip_dataset.json

Each entry contains:

```json
{
  "problem_id": "string",
  "model_name": "string",
  "question": "string",
  "correct_answer": "string",
  "chain_of_thought": "string",
  "final_answer": "string",
  "is_flip": boolean,
  "flip_type": "string",
  "confidence": float,
  "cot_implied_answer": "string",
  "cot_conclusion_sentence": "string",
  "problem_type": "string",
  "problem_subtype": "string",
  "flip_risk": "string"
}
```

### Flip Types

| Type | Description |
|------|-------------|
| `no_flip` | Answer matches CoT reasoning |
| `numeric_reversal` | CoT says A > B, answer says B |
| `calculation_flip` | CoT calculates X, answer says Y |
| `logic_reversal` | CoT concludes yes, answer says no |
| `negation_flip` | Double negation confusion |
| `format_confusion` | Right answer, wrong format |
| `hedging_flip` | CoT confident, answer hedges |
| `apparent_random` | No clear pattern |

## Usage

### Loading in Python

```python
import json

# Load full dataset
with open('flip_dataset.json') as f:
    dataset = json.load(f)

# Filter to flips only
flips = [d for d in dataset if d['is_flip']]
print(f"Found {len(flips)} flips out of {len(dataset)} examples")

# Analyze by problem type
from collections import Counter
flip_types = Counter(d['flip_type'] for d in flips)
print(flip_types)
```

### Loading in Pandas

```python
import pandas as pd

df = pd.read_csv('flip_dataset.csv')
print(df.groupby('problem_type')['is_flip'].mean())
```

## Research Applications

This dataset is useful for:

1. **CoT Faithfulness Research** - Studying when/why models contradict their reasoning
2. **Monitor Development** - Training classifiers to detect unfaithful CoT
3. **Intervention Studies** - Testing methods to prevent flips
4. **Safety Auditing** - Understanding reasoning model failure modes

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{cotflip2024,
  author = {{Amaan, Syed}},
  title = {{CoT Flip Detection Dataset}},
  year = {{2024}},
  url = {{https://github.com/syedamaan/cot-flip-dataset}}
}}
```

## Related Work

- Arcuschin et al. - Unfaithful CoT taxonomy
- Chen et al. - CoT faithfulness measurement  
- Bogdan et al. - Thought anchors paradigm
- Neel Nanda - Pragmatic interpretability framework

## Contact

- Twitter: @syedamaan
- Email: syed@example.com
- Website: syedamaan.com
"""
    
    with open(output_dir / "README.md", 'w') as f:
        f.write(readme)
    
    print(f"Created README: {output_dir / 'README.md'}")

def create_csv_version(json_path: Path, csv_path: Path):
    """Convert JSON dataset to CSV format."""
    with open(json_path) as f:
        data = json.load(f)
    
    if not data:
        print("No data to convert")
        return
    
    # Flatten and select key fields
    fields = [
        'problem_id', 'model_name', 'problem_type', 'problem_subtype',
        'is_flip', 'flip_type', 'confidence',
        'correct_answer', 'final_answer', 'cot_implied_answer',
        'answer_matches_correct', 'cot_matches_correct',
        'cot_quality', 'problem_flip_risk'
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for item in data:
            # Flatten nested fields
            row = {k: item.get(k, '') for k in fields}
            writer.writerow(row)
    
    print(f"Created CSV: {csv_path}")

def package_dataset(
    data_dir: Path,
    output_dir: Path
):
    """Package all dataset components for release."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / "problems").mkdir(exist_ok=True)
    (output_dir / "annotations").mkdir(exist_ok=True)
    (output_dir / "examples").mkdir(exist_ok=True)
    (output_dir / "analysis").mkdir(exist_ok=True)
    (output_dir / "analysis" / "figures").mkdir(exist_ok=True)
    
    # Copy problems
    shutil.copy(
        data_dir / "problems" / "all_problems.json",
        output_dir / "problems" / "all_problems.json"
    )
    
    # Copy annotations
    shutil.copy(
        data_dir / "annotated" / "flip_annotations.json",
        output_dir / "annotations" / "flip_annotations.json"
    )
    
    # Create combined dataset file
    with open(data_dir / "annotated" / "flip_annotations.json") as f:
        annotations = json.load(f)
    
    with open(output_dir / "flip_dataset.json", 'w') as f:
        json.dump(annotations, f, indent=2)
    
    # Create CSV version
    create_csv_version(
        output_dir / "flip_dataset.json",
        output_dir / "flip_dataset.csv"
    )
    
    # Copy examples
    examples_src = data_dir.parent / "outputs" / "flip_examples.json"
    if examples_src.exists():
        shutil.copy(examples_src, output_dir / "examples" / "flip_examples.json")
    
    # Copy analysis
    analysis_src = data_dir.parent / "outputs"
    if analysis_src.exists():
        for f in analysis_src.glob("*.json"):
            shutil.copy(f, output_dir / "analysis" / f.name)
        for f in analysis_src.glob("*.txt"):
            shutil.copy(f, output_dir / "analysis" / f.name)
        for f in analysis_src.glob("*.png"):
            shutil.copy(f, output_dir / "analysis" / "figures" / f.name)
    
    # Load stats for README
    stats_path = data_dir / "annotated" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            raw_stats = json.load(f)
        # Normalize stats structure
        if 'summary' not in raw_stats:
            stats = {
                'summary': {
                    'total_examples': raw_stats.get('total_examples', len(annotations)),
                    'total_flips': raw_stats.get('total_flips', 0),
                    'flip_rate': raw_stats.get('flip_rate', 0),
                    'high_confidence_flips': raw_stats.get('high_confidence_flips', 0),
                },
                'by_flip_type': raw_stats.get('flip_types', {}),
                'by_problem_type': {},
                'accuracy': raw_stats.get('accuracy', {}),
            }
            # Convert flip_rate_by_problem_type to expected format
            for ptype, rate in raw_stats.get('flip_rate_by_problem_type', {}).items():
                stats['by_problem_type'][ptype] = {'total': 0, 'flip_rate': rate}
        else:
            stats = raw_stats
    else:
        stats = {"summary": {"total_examples": len(annotations), "total_flips": 0, "flip_rate": 0, "high_confidence_flips": 0}}
    
    # Create README
    create_readme(output_dir, stats)
    
    print(f"\n{'='*60}")
    print("DATASET PACKAGED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"\nOutput directory: {output_dir}")
    print("\nContents:")
    for item in sorted(output_dir.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(output_dir)
            size = item.stat().st_size
            print(f"  {rel_path} ({size:,} bytes)")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Package dataset for release")
    parser.add_argument("--data-dir", type=str,
                       default="/home/claude/cot-flip-research/data")
    parser.add_argument("--output-dir", type=str,
                       default="/home/claude/cot-flip-research/release/cot-flip-dataset")
    
    args = parser.parse_args()
    
    package_dataset(Path(args.data_dir), Path(args.output_dir))

if __name__ == "__main__":
    main()
