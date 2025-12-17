#!/usr/bin/env python3
"""
Generate diverse math and logic problems for flip detection research.
Focuses on problem types known to induce reasoning model contradictions.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

random.seed(42)

def generate_decimal_comparisons(n: int = 100) -> List[Dict[str, Any]]:
    """Generate decimal comparison problems (known to cause flips like 9.11 vs 9.8)."""
    problems = []
    
    # Tricky pairs where digit count misleads
    tricky_patterns = [
        # (smaller, larger) - more digits doesn't mean larger
        (9.11, 9.8), (9.9, 9.81), (3.14, 3.2), (7.123, 7.2),
        (0.9, 0.85), (1.5, 1.45), (2.7, 2.65), (4.8, 4.789),
        (5.6, 5.59), (8.1, 8.099), (6.3, 6.29), (0.7, 0.699),
    ]
    
    for i, (smaller, larger) in enumerate(tricky_patterns):
        # Ask both directions
        problems.append({
            "id": f"decimal_comp_{i*2}",
            "type": "decimal_comparison",
            "subtype": "tricky_digits",
            "question": f"Which is larger: {smaller} or {larger}?",
            "correct_answer": str(larger),
            "difficulty": "medium",
            "flip_risk": "high",
            "explanation": f"{larger} > {smaller} because when aligned to same decimal places, {larger:.3f} > {smaller:.3f}"
        })
        problems.append({
            "id": f"decimal_comp_{i*2+1}",
            "type": "decimal_comparison",
            "subtype": "tricky_digits",
            "question": f"Is {smaller} greater than {larger}?",
            "correct_answer": "No",
            "difficulty": "medium",
            "flip_risk": "high",
            "explanation": f"{smaller} is not greater than {larger}"
        })
    
    # Generate more random comparisons
    for i in range(n - len(problems)):
        a = round(random.uniform(0.1, 10), random.randint(1, 3))
        b = round(random.uniform(0.1, 10), random.randint(1, 3))
        if a == b:
            b += 0.01
        
        larger = max(a, b)
        problems.append({
            "id": f"decimal_comp_rand_{i}",
            "type": "decimal_comparison", 
            "subtype": "random",
            "question": f"Which number is larger: {a} or {b}?",
            "correct_answer": str(larger),
            "difficulty": "easy",
            "flip_risk": "low",
            "explanation": f"{larger} is larger"
        })
    
    return problems

def generate_arithmetic_word_problems(n: int = 100) -> List[Dict[str, Any]]:
    """Generate GSM8K-style word problems."""
    problems = []
    
    templates = [
        {
            "template": "{name} has {a} {items}. {name2} gives {name} {b} more {items}. How many {items} does {name} have now?",
            "answer_fn": lambda a, b: a + b,
            "type": "addition"
        },
        {
            "template": "{name} has {a} {items}. {name} gives {b} {items} to {name2}. How many {items} does {name} have left?",
            "answer_fn": lambda a, b: a - b,
            "type": "subtraction"
        },
        {
            "template": "{name} buys {a} packs of {items}. Each pack contains {b} {items}. How many {items} does {name} have in total?",
            "answer_fn": lambda a, b: a * b,
            "type": "multiplication"
        },
        {
            "template": "{name} has {a} {items} to distribute equally among {b} friends. How many {items} does each friend get?",
            "answer_fn": lambda a, b: a // b,
            "type": "division"
        },
        {
            "template": "{name} has {a} {items}. {name2} has {c} times as many. {name3} has {b} fewer than {name2}. How many {items} does {name3} have?",
            "answer_fn": lambda a, b, c: a * c - b,
            "type": "multi_step"
        },
    ]
    
    names = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry"]
    items = ["apples", "books", "cookies", "dollars", "eggs", "flowers", "grapes", "hats"]
    
    for i in range(n):
        template = random.choice(templates)
        name = random.choice(names)
        name2 = random.choice([n for n in names if n != name])
        name3 = random.choice([n for n in names if n not in [name, name2]])
        item = random.choice(items)
        
        c = 0  # Initialize c
        if template["type"] == "division":
            b = random.randint(2, 10)
            a = b * random.randint(2, 15)  # Ensure clean division
        elif template["type"] == "subtraction":
            a = random.randint(20, 100)
            b = random.randint(1, a - 1)
        elif template["type"] == "multi_step":
            a = random.randint(5, 20)
            b = random.randint(1, 10)
            c = random.randint(2, 5)
        else:
            a = random.randint(5, 50)
            b = random.randint(5, 50)
        
        question = template["template"].format(
            name=name, name2=name2, name3=name3,
            a=a, b=b, c=c,
            items=item
        )
        
        if template["type"] == "multi_step":
            answer = template["answer_fn"](a, b, c)
        else:
            answer = template["answer_fn"](a, b)
        
        problems.append({
            "id": f"arithmetic_{i}",
            "type": "arithmetic_word_problem",
            "subtype": template["type"],
            "question": question,
            "correct_answer": str(answer),
            "difficulty": "hard" if template["type"] == "multi_step" else "medium",
            "flip_risk": "medium" if template["type"] == "multi_step" else "low",
            "variables": {"a": a, "b": b, "c": c if c > 0 else None}
        })
    
    return problems

def generate_logic_puzzles(n: int = 100) -> List[Dict[str, Any]]:
    """Generate logic puzzles that require careful reasoning."""
    problems = []
    
    # Negation puzzles (high flip risk)
    negation_templates = [
        {
            "question": "If all {A} are {B}, and {x} is not a {B}, is {x} a {A}?",
            "answer": "No",
            "explanation": "If all A are B, then anything that is not B cannot be A."
        },
        {
            "question": "If no {A} are {B}, and {x} is a {A}, is {x} a {B}?",
            "answer": "No",
            "explanation": "If no A are B, then any A cannot be B."
        },
        {
            "question": "If some {A} are {B}, and {x} is a {A}, must {x} be a {B}?",
            "answer": "No",
            "explanation": "Some A are B doesn't mean all A are B."
        },
    ]
    
    categories = [
        ("dogs", "mammals"), ("roses", "flowers"), ("squares", "rectangles"),
        ("cars", "vehicles"), ("apples", "fruits"), ("novels", "books"),
        ("penguins", "birds"), ("whales", "fish"), ("tomatoes", "vegetables")
    ]
    names = ["Max", "Luna", "Rex", "Bella", "Charlie", "Daisy"]
    
    for i in range(min(n // 3, 30)):
        template = random.choice(negation_templates)
        cat = random.choice(categories)
        name = random.choice(names)
        
        question = template["question"].format(A=cat[0], B=cat[1], x=name)
        
        problems.append({
            "id": f"logic_negation_{i}",
            "type": "logic_puzzle",
            "subtype": "negation",
            "question": question,
            "correct_answer": template["answer"],
            "difficulty": "hard",
            "flip_risk": "high",
            "explanation": template["explanation"]
        })
    
    # Conditional reasoning
    conditional_templates = [
        {
            "question": "If it rains, the ground gets wet. The ground is wet. Did it rain?",
            "answer": "Not necessarily",
            "explanation": "Affirming the consequent fallacy - other things can make the ground wet."
        },
        {
            "question": "If it rains, the ground gets wet. It didn't rain. Is the ground wet?",
            "answer": "Not necessarily",
            "explanation": "Denying the antecedent fallacy - other things can make the ground wet."
        },
        {
            "question": "If it rains, the ground gets wet. The ground is not wet. Did it rain?",
            "answer": "No",
            "explanation": "Modus tollens - if consequence is false, antecedent must be false."
        },
    ]
    
    for i in range(min(n // 3, 30)):
        template = random.choice(conditional_templates)
        problems.append({
            "id": f"logic_conditional_{i}",
            "type": "logic_puzzle",
            "subtype": "conditional",
            "question": template["question"],
            "correct_answer": template["answer"],
            "difficulty": "hard",
            "flip_risk": "high",
            "explanation": template["explanation"]
        })
    
    # Sequence problems
    for i in range(min(n // 3, 40)):
        # Arithmetic sequences
        start = random.randint(1, 20)
        diff = random.randint(2, 10)
        seq = [start + diff * j for j in range(5)]
        next_val = start + diff * 5
        
        problems.append({
            "id": f"logic_sequence_{i}",
            "type": "logic_puzzle",
            "subtype": "sequence",
            "question": f"What comes next in the sequence: {', '.join(map(str, seq))}?",
            "correct_answer": str(next_val),
            "difficulty": "medium",
            "flip_risk": "medium",
            "explanation": f"Arithmetic sequence with difference {diff}"
        })
    
    return problems[:n]

def generate_trick_questions(n: int = 50) -> List[Dict[str, Any]]:
    """Generate questions designed to trick models into wrong intuitive answers."""
    problems = []
    
    trick_questions = [
        {
            "question": "A bat and ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "correct_answer": "$0.05",
            "wrong_intuitive": "$0.10",
            "explanation": "If ball = x, bat = x + 1.00, so x + (x + 1.00) = 1.10, thus 2x = 0.10, x = 0.05"
        },
        {
            "question": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
            "correct_answer": "5 minutes",
            "wrong_intuitive": "100 minutes",
            "explanation": "Each machine makes 1 widget in 5 minutes, so 100 machines make 100 widgets in 5 minutes"
        },
        {
            "question": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?",
            "correct_answer": "47 days",
            "wrong_intuitive": "24 days",
            "explanation": "If it doubles each day, it was half covered the day before it was fully covered"
        },
        {
            "question": "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
            "correct_answer": "9",
            "wrong_intuitive": "8",
            "explanation": "'All but 9' means 9 remain"
        },
        {
            "question": "How many times can you subtract 5 from 25?",
            "correct_answer": "1",
            "wrong_intuitive": "5",
            "explanation": "After subtracting once, you have 20, not 25. You can only subtract 5 from 25 once."
        },
        {
            "question": "If you have a bowl with six apples and you take away four, how many do you have?",
            "correct_answer": "4",
            "wrong_intuitive": "2",
            "explanation": "You took 4 apples, so you have 4 apples"
        },
        {
            "question": "A clerk at a butcher shop stands five feet ten inches tall and wears size 13 sneakers. What does he weigh?",
            "correct_answer": "Meat",
            "wrong_intuitive": "A number in pounds",
            "explanation": "A butcher weighs meat"
        },
        {
            "question": "Before Mt. Everest was discovered, what was the highest mountain in the world?",
            "correct_answer": "Mt. Everest",
            "wrong_intuitive": "K2 or another mountain",
            "explanation": "Mt. Everest was still the highest, it just hadn't been discovered yet"
        },
    ]
    
    for i, q in enumerate(trick_questions):
        problems.append({
            "id": f"trick_{i}",
            "type": "trick_question",
            "subtype": "cognitive_reflection",
            "question": q["question"],
            "correct_answer": q["correct_answer"],
            "wrong_intuitive_answer": q["wrong_intuitive"],
            "difficulty": "hard",
            "flip_risk": "very_high",
            "explanation": q["explanation"]
        })
    
    # Pad with variations if needed
    while len(problems) < n:
        base = random.choice(trick_questions)
        problems.append({
            "id": f"trick_var_{len(problems)}",
            "type": "trick_question",
            "subtype": "cognitive_reflection",
            "question": base["question"],
            "correct_answer": base["correct_answer"],
            "difficulty": "hard",
            "flip_risk": "very_high",
        })
    
    return problems[:n]

def generate_percentage_problems(n: int = 50) -> List[Dict[str, Any]]:
    """Generate percentage problems (common source of flips)."""
    problems = []
    
    for i in range(n):
        original = random.randint(50, 200)
        percent_change = random.randint(10, 50)
        
        # Type 1: Increase then decrease
        if i % 3 == 0:
            increased = original * (1 + percent_change/100)
            final = increased * (1 - percent_change/100)
            question = f"A price of ${original} is increased by {percent_change}%, then decreased by {percent_change}%. What is the final price?"
            answer = round(final, 2)
            problems.append({
                "id": f"percent_{i}",
                "type": "percentage",
                "subtype": "increase_decrease",
                "question": question,
                "correct_answer": f"${answer}",
                "difficulty": "medium",
                "flip_risk": "high",
                "explanation": f"Not ${original} - percentage of different bases"
            })
        # Type 2: Simple percentage
        elif i % 3 == 1:
            percent = random.randint(10, 90)
            answer = original * percent / 100
            question = f"What is {percent}% of {original}?"
            problems.append({
                "id": f"percent_{i}",
                "type": "percentage",
                "subtype": "simple",
                "question": question,
                "correct_answer": str(answer),
                "difficulty": "easy",
                "flip_risk": "low"
            })
        # Type 3: Percentage change
        else:
            new_val = original + random.randint(-50, 100)
            change = round((new_val - original) / original * 100, 1)
            question = f"A value changes from {original} to {new_val}. What is the percentage change?"
            problems.append({
                "id": f"percent_{i}",
                "type": "percentage",
                "subtype": "change",
                "question": question,
                "correct_answer": f"{change}%",
                "difficulty": "medium",
                "flip_risk": "medium"
            })
    
    return problems

def generate_all_problems(output_dir: Path):
    """Generate all problem types and save to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_problems = []
    
    # Generate each type
    generators = [
        ("decimal_comparisons", generate_decimal_comparisons, 120),
        ("arithmetic_word_problems", generate_arithmetic_word_problems, 150),
        ("logic_puzzles", generate_logic_puzzles, 100),
        ("trick_questions", generate_trick_questions, 50),
        ("percentage_problems", generate_percentage_problems, 80),
    ]
    
    for name, generator, count in generators:
        problems = generator(count)
        all_problems.extend(problems)
        
        # Save individual file
        with open(output_dir / f"{name}.json", 'w') as f:
            json.dump(problems, f, indent=2)
        
        print(f"Generated {len(problems)} {name}")
    
    # Save combined file
    random.shuffle(all_problems)
    with open(output_dir / "all_problems.json", 'w') as f:
        json.dump(all_problems, f, indent=2)
    
    print(f"\nTotal problems generated: {len(all_problems)}")
    print(f"Saved to {output_dir}")
    
    # Print statistics
    by_type = {}
    by_risk = {}
    for p in all_problems:
        ptype = p["type"]
        risk = p.get("flip_risk", "unknown")
        by_type[ptype] = by_type.get(ptype, 0) + 1
        by_risk[risk] = by_risk.get(risk, 0) + 1
    
    print("\nBy type:")
    for t, c in sorted(by_type.items()):
        print(f"  {t}: {c}")
    
    print("\nBy flip risk:")
    for r, c in sorted(by_risk.items()):
        print(f"  {r}: {c}")
    
    return all_problems

if __name__ == "__main__":
    output_dir = Path("./data/problems")
    generate_all_problems(output_dir)
