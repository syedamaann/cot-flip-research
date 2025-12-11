#!/usr/bin/env python3
"""
Comprehensive analysis and visualization of flip detection results.
Produces publication-ready charts and statistics.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np

# Style settings for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def load_annotations(path: Path) -> List[Dict]:
    """Load annotated data."""
    with open(path) as f:
        return json.load(f)

def calculate_detailed_stats(annotations: List[Dict]) -> Dict[str, Any]:
    """Calculate comprehensive statistics."""
    
    total = len(annotations)
    flips = [a for a in annotations if a["is_flip"]]
    non_flips = [a for a in annotations if not a["is_flip"]]
    
    stats = {
        "summary": {
            "total_examples": total,
            "total_flips": len(flips),
            "total_non_flips": len(non_flips),
            "flip_rate": len(flips) / total if total > 0 else 0,
            "high_confidence_flips": sum(1 for a in flips if a["confidence"] >= 0.8),
        },
        "by_flip_type": dict(Counter(a["flip_type"] for a in flips)),
        "by_problem_type": {},
        "by_risk_level": {},
        "by_cot_quality": {},
        "accuracy": {},
        "confidence_distribution": {
            "mean": np.mean([a["confidence"] for a in flips]) if flips else 0,
            "std": np.std([a["confidence"] for a in flips]) if flips else 0,
            "bins": np.histogram([a["confidence"] for a in flips], bins=10)[0].tolist() if flips else [],
        },
    }
    
    # By problem type
    by_ptype = defaultdict(lambda: {"total": 0, "flips": 0, "correct": 0})
    for a in annotations:
        ptype = a.get("problem_type", "unknown")
        by_ptype[ptype]["total"] += 1
        if a["is_flip"]:
            by_ptype[ptype]["flips"] += 1
        if a.get("answer_matches_correct", False):
            by_ptype[ptype]["correct"] += 1
    
    stats["by_problem_type"] = {
        k: {
            "total": v["total"],
            "flips": v["flips"],
            "flip_rate": v["flips"] / v["total"] if v["total"] > 0 else 0,
            "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0,
        }
        for k, v in by_ptype.items()
    }
    
    # By risk level
    by_risk = defaultdict(lambda: {"total": 0, "flips": 0})
    for a in annotations:
        risk = a.get("problem_flip_risk", "unknown")
        by_risk[risk]["total"] += 1
        if a["is_flip"]:
            by_risk[risk]["flips"] += 1
    
    stats["by_risk_level"] = {
        k: {
            "total": v["total"],
            "flips": v["flips"],
            "flip_rate": v["flips"] / v["total"] if v["total"] > 0 else 0,
        }
        for k, v in by_risk.items()
    }
    
    # By CoT quality
    by_quality = defaultdict(lambda: {"total": 0, "flips": 0})
    for a in annotations:
        quality = a.get("cot_quality", "unknown")
        by_quality[quality]["total"] += 1
        if a["is_flip"]:
            by_quality[quality]["flips"] += 1
    
    stats["by_cot_quality"] = {
        k: {
            "total": v["total"],
            "flips": v["flips"],
            "flip_rate": v["flips"] / v["total"] if v["total"] > 0 else 0,
        }
        for k, v in by_quality.items()
    }
    
    # Accuracy analysis
    stats["accuracy"] = {
        "overall": sum(1 for a in annotations if a.get("answer_matches_correct")) / total if total else 0,
        "flip_answer_correct": sum(1 for a in flips if a.get("answer_matches_correct")) / len(flips) if flips else 0,
        "flip_cot_correct": sum(1 for a in flips if a.get("cot_matches_correct")) / len(flips) if flips else 0,
        "non_flip_correct": sum(1 for a in non_flips if a.get("answer_matches_correct")) / len(non_flips) if non_flips else 0,
    }
    
    return stats

def plot_flip_rate_by_problem_type(stats: Dict, output_path: Path):
    """Bar chart of flip rates by problem type."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data = stats["by_problem_type"]
    types = sorted(data.keys(), key=lambda x: data[x]["flip_rate"], reverse=True)
    rates = [data[t]["flip_rate"] * 100 for t in types]
    counts = [data[t]["total"] for t in types]
    
    # Create bars
    bars = ax.bar(range(len(types)), rates, color='steelblue', edgecolor='navy', alpha=0.8)
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'n={count}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([t.replace('_', '\n') for t in types], fontsize=10)
    ax.set_ylabel('Flip Rate (%)')
    ax.set_xlabel('Problem Type')
    ax.set_title('Last-Minute Answer Flip Rate by Problem Type')
    ax.set_ylim(0, max(rates) * 1.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_flip_type_distribution(stats: Dict, output_path: Path):
    """Pie chart of flip type distribution."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    data = stats["by_flip_type"]
    if not data:
        print("No flip data to plot")
        return
    
    labels = list(data.keys())
    sizes = list(data.values())
    
    # Colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    # Create pie
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=90,
        explode=[0.02] * len(labels)
    )
    
    # Styling
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_color('black')
    
    ax.set_title('Distribution of Flip Types')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_flip_rate_by_risk(stats: Dict, output_path: Path):
    """Bar chart comparing flip rate to expected risk level."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = stats["by_risk_level"]
    
    # Order risk levels
    risk_order = ["low", "medium", "high", "very_high"]
    risks = [r for r in risk_order if r in data]
    rates = [data[r]["flip_rate"] * 100 for r in risks]
    counts = [data[r]["total"] for r in risks]
    
    x = range(len(risks))
    bars = ax.bar(x, rates, color=['green', 'yellow', 'orange', 'red'][:len(risks)], 
                  edgecolor='black', alpha=0.7)
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'n={count}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace('_', ' ').title() for r in risks])
    ax.set_ylabel('Observed Flip Rate (%)')
    ax.set_xlabel('Expected Risk Level')
    ax.set_title('Flip Rate vs Expected Risk Level')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_accuracy_comparison(stats: Dict, output_path: Path):
    """Compare accuracy between flip and non-flip cases."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    acc = stats["accuracy"]
    
    categories = ['Overall\nAccuracy', 'Flip Cases\n(Answer)', 'Flip Cases\n(CoT Would Be)', 'Non-Flip\nCases']
    values = [
        acc["overall"] * 100,
        acc["flip_answer_correct"] * 100,
        acc["flip_cot_correct"] * 100,
        acc["non_flip_correct"] * 100,
    ]
    
    colors = ['steelblue', 'coral', 'lightgreen', 'steelblue']
    bars = ax.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Analysis: Impact of Flips')
    ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_cot_quality_vs_flips(stats: Dict, output_path: Path):
    """Compare flip rates across CoT quality levels."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = stats["by_cot_quality"]
    
    qualities = ["poor", "mixed", "good"]
    qualities = [q for q in qualities if q in data]
    rates = [data[q]["flip_rate"] * 100 for q in qualities]
    counts = [data[q]["total"] for q in qualities]
    
    bars = ax.bar(qualities, rates, color='mediumpurple', edgecolor='purple', alpha=0.8)
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'n={count}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Flip Rate (%)')
    ax.set_xlabel('Chain of Thought Quality')
    ax.set_title('Flip Rate by Chain of Thought Quality')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_summary_dashboard(stats: Dict, output_path: Path):
    """Create a summary dashboard with key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Overall stats (text)
    ax1 = axes[0, 0]
    ax1.axis('off')
    summary = stats["summary"]
    text = f"""
    DATASET SUMMARY
    ─────────────────────────
    Total Examples:     {summary['total_examples']}
    Total Flips:        {summary['total_flips']}
    Overall Flip Rate:  {summary['flip_rate']:.1%}
    High-Confidence:    {summary['high_confidence_flips']}
    
    ACCURACY IMPACT
    ─────────────────────────
    Overall Accuracy:         {stats['accuracy']['overall']:.1%}
    Flip Answer Correct:      {stats['accuracy']['flip_answer_correct']:.1%}
    Flip CoT Would Be Correct:{stats['accuracy']['flip_cot_correct']:.1%}
    Non-Flip Accuracy:        {stats['accuracy']['non_flip_correct']:.1%}
    """
    ax1.text(0.1, 0.5, text, fontsize=12, family='monospace',
             verticalalignment='center', transform=ax1.transAxes)
    ax1.set_title('Key Statistics', fontsize=14, fontweight='bold')
    
    # 2. Flip types pie chart
    ax2 = axes[0, 1]
    data = stats["by_flip_type"]
    if data:
        labels = list(data.keys())
        sizes = list(data.values())
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels)))
        ax2.pie(sizes, labels=labels, autopct='%1.0f%%', colors=colors, startangle=90)
    ax2.set_title('Flip Type Distribution', fontsize=14, fontweight='bold')
    
    # 3. Flip rate by problem type
    ax3 = axes[1, 0]
    ptype_data = stats["by_problem_type"]
    types = sorted(ptype_data.keys(), key=lambda x: ptype_data[x]["flip_rate"], reverse=True)[:5]
    rates = [ptype_data[t]["flip_rate"] * 100 for t in types]
    ax3.barh(range(len(types)), rates, color='steelblue', alpha=0.8)
    ax3.set_yticks(range(len(types)))
    ax3.set_yticklabels([t.replace('_', ' ') for t in types])
    ax3.set_xlabel('Flip Rate (%)')
    ax3.set_title('Top 5 Problem Types by Flip Rate', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()
    
    # 4. Risk level comparison
    ax4 = axes[1, 1]
    risk_data = stats["by_risk_level"]
    risk_order = ["low", "medium", "high", "very_high"]
    risks = [r for r in risk_order if r in risk_data]
    risk_rates = [risk_data[r]["flip_rate"] * 100 for r in risks]
    colors = ['green', 'yellow', 'orange', 'red'][:len(risks)]
    ax4.bar(range(len(risks)), risk_rates, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_xticks(range(len(risks)))
    ax4.set_xticklabels([r.replace('_', ' ').title() for r in risks])
    ax4.set_ylabel('Flip Rate (%)')
    ax4.set_title('Flip Rate by Risk Level', fontsize=14, fontweight='bold')
    
    plt.suptitle('CoT Flip Detection Analysis Dashboard', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def export_flip_examples(annotations: List[Dict], output_path: Path, n: int = 20):
    """Export representative flip examples for manual review."""
    flips = [a for a in annotations if a["is_flip"]]
    
    # Select diverse examples
    by_type = defaultdict(list)
    for f in flips:
        by_type[f["flip_type"]].append(f)
    
    examples = []
    for flip_type, items in by_type.items():
        # Take up to n/num_types from each type
        n_per_type = max(1, n // len(by_type))
        examples.extend(items[:n_per_type])
    
    # Truncate to n
    examples = examples[:n]
    
    # Format for review
    formatted = []
    for i, ex in enumerate(examples):
        formatted.append({
            "index": i + 1,
            "problem_id": ex["problem_id"],
            "flip_type": ex["flip_type"],
            "confidence": ex["confidence"],
            "problem_type": ex.get("problem_type", ""),
            "question": ex.get("final_answer", ""),  # This would be the question in real data
            "chain_of_thought": ex.get("chain_of_thought", "")[:500] + "..." if len(ex.get("chain_of_thought", "")) > 500 else ex.get("chain_of_thought", ""),
            "cot_conclusion": ex["cot_conclusion_sentence"],
            "cot_implied_answer": ex["cot_implied_answer"],
            "final_answer": ex["final_answer"],
            "correct_answer": ex["correct_answer"],
        })
    
    with open(output_path, 'w') as f:
        json.dump(formatted, f, indent=2)
    
    print(f"Exported {len(formatted)} flip examples to: {output_path}")

def generate_report(stats: Dict, output_path: Path):
    """Generate a text report of findings."""
    summary = stats["summary"]
    
    report = f"""
================================================================================
                    COT FLIP DETECTION ANALYSIS REPORT
================================================================================

EXECUTIVE SUMMARY
-----------------
This analysis examines {summary['total_examples']} reasoning model outputs to detect
"last-minute answer flips" - cases where the chain of thought implies one answer
but the model outputs a different final answer.

KEY FINDINGS
------------
• Total examples analyzed: {summary['total_examples']}
• Flips detected: {summary['total_flips']} ({summary['flip_rate']:.1%} of all examples)
• High-confidence flips: {summary['high_confidence_flips']}

FLIP TYPE DISTRIBUTION
----------------------
"""
    for ftype, count in sorted(stats["by_flip_type"].items(), key=lambda x: -x[1]):
        pct = count / summary['total_flips'] * 100 if summary['total_flips'] > 0 else 0
        report += f"• {ftype}: {count} ({pct:.1f}%)\n"
    
    report += """
FLIP RATE BY PROBLEM TYPE
-------------------------
"""
    for ptype, data in sorted(stats["by_problem_type"].items(), 
                               key=lambda x: -x[1]["flip_rate"]):
        report += f"• {ptype}: {data['flip_rate']:.1%} (n={data['total']})\n"
    
    report += """
FLIP RATE BY RISK LEVEL
-----------------------
"""
    risk_order = ["low", "medium", "high", "very_high"]
    for risk in risk_order:
        if risk in stats["by_risk_level"]:
            data = stats["by_risk_level"][risk]
            report += f"• {risk}: {data['flip_rate']:.1%} (n={data['total']})\n"
    
    report += f"""
ACCURACY ANALYSIS
-----------------
• Overall accuracy: {stats['accuracy']['overall']:.1%}
• Flip cases - answer correct: {stats['accuracy']['flip_answer_correct']:.1%}
• Flip cases - CoT would be correct: {stats['accuracy']['flip_cot_correct']:.1%}
• Non-flip accuracy: {stats['accuracy']['non_flip_correct']:.1%}

IMPLICATIONS
------------
1. Flips occur at a rate of {summary['flip_rate']:.1%}, which is {'significant' if summary['flip_rate'] > 0.1 else 'relatively low'}.
2. {'The CoT often gives correct reasoning that is contradicted by the final answer.' if stats['accuracy']['flip_cot_correct'] > stats['accuracy']['flip_answer_correct'] else 'Flips do not always lead to worse answers.'}
3. Problem types with highest flip rates should be prioritized for intervention.

================================================================================
                              END OF REPORT
================================================================================
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {output_path}")
    return report

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze flip detection results")
    parser.add_argument("--input", type=str,
                       default="/home/claude/cot-flip-research/data/annotated/flip_annotations.json")
    parser.add_argument("--output-dir", type=str,
                       default="/home/claude/cot-flip-research/outputs")
    
    args = parser.parse_args()
    
    # Load data
    annotations = load_annotations(Path(args.input))
    print(f"Loaded {len(annotations)} annotations")
    
    # Calculate stats
    stats = calculate_detailed_stats(annotations)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all outputs
    plot_flip_rate_by_problem_type(stats, output_dir / "flip_rate_by_problem_type.png")
    plot_flip_type_distribution(stats, output_dir / "flip_type_distribution.png")
    plot_flip_rate_by_risk(stats, output_dir / "flip_rate_by_risk.png")
    plot_accuracy_comparison(stats, output_dir / "accuracy_comparison.png")
    plot_cot_quality_vs_flips(stats, output_dir / "cot_quality_vs_flips.png")
    plot_summary_dashboard(stats, output_dir / "summary_dashboard.png")
    
    # Export examples and report
    export_flip_examples(annotations, output_dir / "flip_examples.json")
    report = generate_report(stats, output_dir / "analysis_report.txt")
    
    # Save detailed stats
    with open(output_dir / "detailed_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()
