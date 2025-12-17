#!/usr/bin/env python3
"""
Collect reasoning model outputs for flip detection research.
Supports multiple inference backends: vLLM, OpenAI API, Anthropic API, local transformers.
"""

import json
import time
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm
import random

@dataclass
class ModelOutput:
    """Represents a single model response."""
    problem_id: str
    model_name: str
    question: str
    correct_answer: str
    
    # Model's response
    full_response: str
    chain_of_thought: str
    final_answer: str
    
    # Metadata
    timestamp: str
    inference_time_ms: float
    token_count: Optional[int] = None
    
    # Annotations (filled later)
    cot_implies_answer: Optional[str] = None
    is_flip: Optional[bool] = None
    flip_type: Optional[str] = None
    notes: Optional[str] = None

def extract_cot_and_answer(response: str) -> Tuple[str, str]:
    """
    Extract chain of thought and final answer from model response.
    Handles various response formats including DeepSeek-R1's format.

    Priority order:
    1. LaTeX \boxed{} format (most reliable for math answers)
    2. </think> tag format (DeepSeek-R1 reasoning format)
    3. Standard answer markers
    4. Last line fallback
    """
    response = response.strip()
    chain_of_thought = ""
    final_answer = ""

    # Priority 1: Look for \boxed{} LaTeX format (find the LAST occurrence)
    boxed_matches = list(re.finditer(r'\\boxed\{([^}]+)\}', response))
    if boxed_matches:
        last_match = boxed_matches[-1]
        final_answer = last_match.group(1).strip()
        # Everything before the last \boxed is CoT
        chain_of_thought = response[:last_match.start()].strip()
        return chain_of_thought, final_answer

    # Priority 2: Look for </think> tag (DeepSeek-R1 format)
    think_match = re.search(r'</think>\s*(.*)$', response, re.DOTALL | re.IGNORECASE)
    if think_match:
        chain_of_thought = response[:response.rfind('</think>')].strip()
        # Remove <think> tag if present at start
        chain_of_thought = re.sub(r'^\s*<think>\s*', '', chain_of_thought, flags=re.IGNORECASE)

        final_section = think_match.group(1).strip()

        # Try to extract answer from the final section
        answer_patterns = [
            r'\*\*Answer:\*\*\s*(.+?)(?:\n|$)',
            r'Answer:\s*(.+?)(?:\n|$)',
            r'\*\*Final Answer\*\*[:\s]*(.+?)(?:\n|$)',
            r'Final Answer[:\s]*(.+?)(?:\n|$)',
            r'(?:Therefore|Thus|So),?\s+(?:the answer is\s+)?(.+?)(?:\.|$)',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, final_section, re.IGNORECASE)
            if match:
                final_answer = match.group(1).strip()
                break

        # If still no answer, take first substantial line from final section
        if not final_answer and final_section:
            lines = [l.strip() for l in final_section.split('\n') if l.strip()]
            if lines:
                final_answer = lines[0]

        return chain_of_thought, final_answer

    # Priority 3: Standard answer markers
    patterns = [
        r'\*\*Answer:\*\*\s*(.+?)(?:\n|$)',
        r'Answer:\s*(.+?)(?:\n|$)',
        r'\*\*Final Answer\*\*[:\s]*(.+?)(?:\n|$)',
        r'Final Answer[:\s]*(.+?)(?:\n|$)',
        r'The answer is[:\s]*(.+?)(?:\n|$)',
        r'Therefore, the answer is[:\s]*(.+?)(?:\n|$)',
        r'(?:\*\*Answer\*\*|###\s*Answer)[:\s]*(.+?)(?:\n|$)',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            final_answer = match.group(1).strip()
            chain_of_thought = response[:match.start()].strip()
            return chain_of_thought, final_answer

    # Priority 4: Last line fallback (when no markers found)
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    if lines:
        final_answer = lines[-1]
        # Clean up common prefixes
        for prefix in ["So ", "Thus ", "Therefore ", "Hence ", "The answer is ", "Answer: "]:
            if final_answer.lower().startswith(prefix.lower()):
                final_answer = final_answer[len(prefix):].strip()
                break

        # CoT is everything except last paragraph
        paragraphs = response.split('\n\n')
        if len(paragraphs) > 1:
            chain_of_thought = '\n\n'.join(paragraphs[:-1])
        else:
            chain_of_thought = response

    # Clean up markdown artifacts from final answer
    final_answer = re.sub(r'^\*+\s*|\s*\*+$', '', final_answer)  # Remove leading/trailing asterisks
    final_answer = re.sub(r'^\.+\s*|\s*\.+$', '', final_answer)  # Remove standalone dots

    return chain_of_thought, final_answer

def create_prompt(question: str, include_cot_instruction: bool = True) -> str:
    """Create the prompt for the reasoning model."""
    if include_cot_instruction:
        return f"""Solve this problem step by step, showing your reasoning clearly.

Question: {question}

Think through this carefully, then provide your final answer."""
    else:
        return f"Question: {question}"

class InferenceBackend:
    """Base class for inference backends."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def generate(self, prompt: str, max_tokens: int = 2048) -> Tuple[str, float, Optional[int]]:
        """Generate response. Returns (response, time_ms, token_count)."""
        raise NotImplementedError

class OpenAIBackend(InferenceBackend):
    """OpenAI API backend (works with compatible APIs like DeepSeek)."""
    
    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None):
        super().__init__(model_name)
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def generate(self, prompt: str, max_tokens: int = 2048) -> Tuple[str, float, Optional[int]]:
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        elapsed = (time.time() - start) * 1000
        
        text = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else None
        
        return text, elapsed, tokens

class AnthropicBackend(InferenceBackend):
    """Anthropic API backend."""
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
    
    def generate(self, prompt: str, max_tokens: int = 2048) -> Tuple[str, float, Optional[int]]:
        start = time.time()
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        elapsed = (time.time() - start) * 1000
        
        text = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        
        return text, elapsed, tokens

class VLLMBackend(InferenceBackend):
    """vLLM backend for local inference."""

    def __init__(self, model_name: str, tensor_parallel_size: int = 1):
        super().__init__(model_name)
        try:
            from vllm import LLM, SamplingParams
            self.llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)
            self.SamplingParams = SamplingParams
        except ImportError:
            raise ImportError("Please install vllm: pip install vllm")

    def generate(self, prompt: str, max_tokens: int = 2048) -> Tuple[str, float, Optional[int]]:
        sampling_params = self.SamplingParams(
            temperature=0.7,
            max_tokens=max_tokens,
        )

        start = time.time()
        outputs = self.llm.generate([prompt], sampling_params)
        elapsed = (time.time() - start) * 1000

        text = outputs[0].outputs[0].text
        tokens = len(outputs[0].outputs[0].token_ids)

        return text, elapsed, tokens

    def generate_batch(self, prompts: List[str], max_tokens: int = 2048) -> List[Tuple[str, float, int]]:
        """
        Generate responses for a batch of prompts efficiently.
        Returns list of (response, time_ms, token_count) tuples.
        """
        sampling_params = self.SamplingParams(
            temperature=0.7,
            max_tokens=max_tokens,
        )

        start = time.time()
        outputs = self.llm.generate(prompts, sampling_params)
        total_elapsed = (time.time() - start) * 1000

        # Distribute time across batch (approximation)
        time_per_prompt = total_elapsed / len(prompts)

        results = []
        for output in outputs:
            text = output.outputs[0].text
            tokens = len(output.outputs[0].token_ids)
            results.append((text, time_per_prompt, tokens))

        return results

class TransformersBackend(InferenceBackend):
    """HuggingFace Transformers backend."""
    
    def __init__(self, model_name: str, device: str = "auto"):
        super().__init__(model_name)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True
            )
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers torch")
    
    def generate(self, prompt: str, max_tokens: int = 2048) -> Tuple[str, float, Optional[int]]:
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        elapsed = (time.time() - start) * 1000
        
        # Decode only new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return text, elapsed, len(new_tokens)

class MockBackend(InferenceBackend):
    """Mock backend for testing the pipeline without real inference."""
    
    def __init__(self, model_name: str = "mock-model"):
        super().__init__(model_name)
        self.flip_rate = 0.15  # Simulate 15% flip rate
    
    def generate(self, prompt: str, max_tokens: int = 2048) -> Tuple[str, float, Optional[int]]:
        import time
        time.sleep(0.1)  # Simulate latency
        
        # Extract the question
        question = prompt.split("Question:")[-1].split("Think through")[0].strip()
        
        # Generate mock reasoning
        should_flip = random.random() < self.flip_rate
        
        # Create mock response based on question type
        if "larger" in question.lower() or "greater" in question.lower():
            # Decimal comparison
            numbers = re.findall(r'\d+\.?\d*', question)
            if len(numbers) >= 2:
                a, b = float(numbers[0]), float(numbers[1])
                correct = str(max(a, b))
                wrong = str(min(a, b))
                
                if should_flip:
                    response = f"""Let me compare these numbers carefully.

{a} has {len(str(a).split('.')[-1]) if '.' in str(a) else 0} decimal places.
{b} has {len(str(b).split('.')[-1]) if '.' in str(b) else 0} decimal places.

To compare properly, I'll align the decimal places:
{a} = {a:.3f}
{b} = {b:.3f}

Looking at these values, {correct} is larger because {correct:.3f} > {wrong:.3f}.

Final Answer: {wrong}"""
                else:
                    response = f"""Let me compare these numbers carefully.

{a} has {len(str(a).split('.')[-1]) if '.' in str(a) else 0} decimal places.
{b} has {len(str(b).split('.')[-1]) if '.' in str(b) else 0} decimal places.

To compare properly:
{a} = {a:.3f}
{b} = {b:.3f}

Comparing: {max(a,b):.3f} > {min(a,b):.3f}

Final Answer: {correct}"""
            else:
                response = "I cannot parse the numbers in this question."
        
        elif "?" in question:
            # Generic question - generate simple mock response
            if should_flip:
                response = """Let me think through this step by step.

First, I'll analyze the problem...
The key insight here is that we need to consider all factors.
After careful reasoning, the answer should be X.

However, upon reflection, I believe the answer is Y.

Final Answer: Y"""
            else:
                response = """Let me think through this step by step.

First, I'll analyze the problem...
The key insight is that we need to apply the correct approach.
Working through the logic carefully...

Final Answer: The answer follows from the reasoning above."""
        else:
            response = "I'm not sure how to parse this question format."
        
        return response, 100.0, len(response.split())

def collect_data(
    problems: List[Dict[str, Any]],
    backend: InferenceBackend,
    output_file: Path,
    checkpoint_every: int = 10,
    max_problems: Optional[int] = None,
    batch_size: int = 1,
) -> List[ModelOutput]:
    """
    Collect model outputs for all problems.
    Saves checkpoints periodically.
    Supports batch processing for faster inference with vLLM.
    """
    results = []

    # Load existing checkpoint if available
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
            results = [ModelOutput(**r) for r in existing]
            completed_ids = {r.problem_id for r in results}
            print(f"Loaded {len(results)} existing results")
    else:
        completed_ids = set()

    # Filter to uncompleted problems
    remaining = [p for p in problems if p["id"] not in completed_ids]
    if max_problems:
        remaining = remaining[:max_problems]

    print(f"Processing {len(remaining)} problems with {backend.model_name}")
    if batch_size > 1:
        print(f"Using batch size: {batch_size}")

    # Check if backend supports batch processing
    supports_batch = hasattr(backend, 'generate_batch') and batch_size > 1

    if supports_batch:
        # Batch processing mode
        for batch_start in tqdm(range(0, len(remaining), batch_size), desc="Batches"):
            batch_problems = remaining[batch_start:batch_start + batch_size]

            try:
                # Create prompts for batch
                prompts = [create_prompt(p["question"]) for p in batch_problems]

                # Generate batch
                batch_results = backend.generate_batch(prompts)

                # Process each result in batch
                for problem, (response, time_ms, tokens) in zip(batch_problems, batch_results):
                    cot, answer = extract_cot_and_answer(response)

                    output = ModelOutput(
                        problem_id=problem["id"],
                        model_name=backend.model_name,
                        question=problem["question"],
                        correct_answer=problem["correct_answer"],
                        full_response=response,
                        chain_of_thought=cot,
                        final_answer=answer,
                        timestamp=datetime.now().isoformat(),
                        inference_time_ms=time_ms,
                        token_count=tokens,
                    )
                    results.append(output)

                # Checkpoint
                if len(results) % checkpoint_every == 0:
                    save_results(results, output_file)
                    print(f"\nCheckpoint saved: {len(results)} results")

            except Exception as e:
                print(f"\nError on batch starting at {batch_start}: {e}")
                # Fall back to individual processing for this batch
                for problem in batch_problems:
                    try:
                        prompt = create_prompt(problem["question"])
                        response, time_ms, tokens = backend.generate(prompt)
                        cot, answer = extract_cot_and_answer(response)

                        output = ModelOutput(
                            problem_id=problem["id"],
                            model_name=backend.model_name,
                            question=problem["question"],
                            correct_answer=problem["correct_answer"],
                            full_response=response,
                            chain_of_thought=cot,
                            final_answer=answer,
                            timestamp=datetime.now().isoformat(),
                            inference_time_ms=time_ms,
                            token_count=tokens,
                        )
                        results.append(output)
                    except Exception as e2:
                        print(f"Error on problem {problem['id']}: {e2}")
                        continue
    else:
        # Sequential processing mode
        for i, problem in enumerate(tqdm(remaining, desc="Problems")):
            try:
                prompt = create_prompt(problem["question"])
                response, time_ms, tokens = backend.generate(prompt)

                cot, answer = extract_cot_and_answer(response)

                output = ModelOutput(
                    problem_id=problem["id"],
                    model_name=backend.model_name,
                    question=problem["question"],
                    correct_answer=problem["correct_answer"],
                    full_response=response,
                    chain_of_thought=cot,
                    final_answer=answer,
                    timestamp=datetime.now().isoformat(),
                    inference_time_ms=time_ms,
                    token_count=tokens,
                )

                results.append(output)

                # Checkpoint
                if (i + 1) % checkpoint_every == 0:
                    save_results(results, output_file)
                    print(f"\nCheckpoint saved: {len(results)} results")

            except Exception as e:
                print(f"\nError on problem {problem['id']}: {e}")
                continue

    # Final save
    save_results(results, output_file)
    print(f"\nCollection complete: {len(results)} total results")

    return results

def save_results(results: List[ModelOutput], output_file: Path):
    """Save results to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

def load_results(input_file: Path) -> List[ModelOutput]:
    """Load results from JSON file."""
    with open(input_file) as f:
        data = json.load(f)
    return [ModelOutput(**r) for r in data]

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Collect reasoning model outputs")
    parser.add_argument("--backend", choices=["openai", "anthropic", "vllm", "transformers", "mock"],
                       default="vllm", help="Inference backend to use")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                       help="Model name/path")
    parser.add_argument("--api-key", type=str, help="API key (for openai/anthropic backends)")
    parser.add_argument("--base-url", type=str, help="Base URL (for openai-compatible APIs)")
    parser.add_argument("--problems", type=str,
                       default="./data/problems/all_problems.json",
                       help="Path to problems JSON")
    parser.add_argument("--output", type=str,
                       default="./data/raw/model_outputs.json",
                       help="Output file path")
    parser.add_argument("--max", type=int, help="Maximum number of problems to process")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for inference (vLLM supports batching). Recommended: 8-32")

    args = parser.parse_args()

    # Initialize backend
    if args.backend == "mock":
        backend = MockBackend(args.model)
    elif args.backend == "openai":
        backend = OpenAIBackend(args.model, args.api_key, args.base_url)
    elif args.backend == "anthropic":
        backend = AnthropicBackend(args.model, args.api_key)
    elif args.backend == "vllm":
        backend = VLLMBackend(args.model)
    elif args.backend == "transformers":
        backend = TransformersBackend(args.model)

    # Load problems
    with open(args.problems) as f:
        problems = json.load(f)

    print(f"Loaded {len(problems)} problems")

    # Collect data
    results = collect_data(
        problems=problems,
        backend=backend,
        output_file=Path(args.output),
        checkpoint_every=args.checkpoint_every,
        max_problems=args.max,
        batch_size=args.batch_size,
    )

    print(f"\nCollection complete!")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
