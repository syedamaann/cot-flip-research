# Understanding Last-Minute Answer Flips in Reasoning Models.

## The Phenomenon
Reasoning models (DeepSeek-R1, QwQ, o1-like models) sometimes exhibit a peculiar failure mode: they produce a coherent, reasonable chain of thought that points toward answer A, then at the final moment, output answer B instead. This "last-minute flip" is one of several forms of unfaithful chain of thought identified in recent work (Arcuschin et al., Chen et al.).

### Example
**User:** Is 9.11 greater than 9.8?

**Model's CoT:** 
> Let me compare these decimals. 9.11 has two decimal places while 9.8 has one. To compare properly, I can write 9.8 as 9.80. Now comparing 9.11 and 9.80... 9.80 is larger because 80 > 11.

**Final Answer:** Yes, 9.11 is greater than 9.8.

*The reasoning was correct, but the final answer contradicts it.*

## Research Questions
1. **Characterization:** How often do last-minute flips occur? On what types of problems? In which models?
2. **Prediction:** Can we identify sentences in the CoT that predict a flip will happen? Are there "warning signs" before the flip?
3. **Mechanism:** What causes flips? Hypotheses include:
    - Preconceived answers overriding reasoning
    - Attention to question phrasing at final token generation
    - Competing "intuition" vs "reasoning" circuits
    - Format/template effects from training
4. **Intervention:** Can we prevent flips? Through:
    - Prompt modifications
    - Steering vectors
    - Resampling specific sentences
    - Editing the CoT before final answer generation

## Why This Matters (North Star)
If we want to trust reasoning models' chains of thought as a safety mechanism ("just read what they're thinking"), we need to know when the CoT actually drives the answer. Last-minute flips are a clean, observable case where CoT is demonstrably unfaithful. Understanding this specific failure mode teaches us:
- How to detect unfaithful reasoning
- What makes CoT more or less reliable
- How to build monitors for reasoning quality

## Proxy Tasks
- **Detection task:** Given a (CoT, answer) pair, predict whether the answer matches what the CoT implies
- **Prediction task:** Given a partial CoT (before the final answer), predict whether a flip will occur
- **Intervention task:** Given a CoT that would flip, modify it minimally to prevent the flip
- **Causal task:** Identify which sentence(s) in the CoT, if removed or edited, change whether a flip occurs

## Technical Approach

### Models
- **Primary:** DeepSeek-R1-Distill-Qwen-7B (open weights, fits on consumer GPU)
- **Validation:** DeepSeek-R1-Distill-Qwen-14B, QwQ-32B-Preview

### Datasets
- Math problems (GSM8K, MATH) where ground truth is known
- Logic puzzles with verifiable answers
- Comparison tasks (like the 9.11 vs 9.8 example)
- Custom adversarial examples designed to induce flips

### Methods (in order of simplicity)
1. Reading CoTs and manually categorizing flip types
2. LLM-as-judge to detect CoT/answer inconsistency
3. Logit analysis at the final answer token
4. Attention pattern analysis (what does model attend to when generating final answer?)
5. Probing for "intended answer" throughout the CoT
6. Steering/ablation experiments

## Success Criteria
A successful project will:
- Produce a dataset of 500+ flip examples with annotations
- Identify at least 2-3 predictive features of flips
- Demonstrate at least one intervention that reduces flip rate
- Provide a falsifiable hypothesis about the mechanism

## Repository Structure
- `generate_problems.py`: Creates 500 diverse math/logic problems across 5 categories
- `collect_data.py`: Runs inference with reasoning models (supports OpenAI API, Anthropic, vLLM, HuggingFace)
- `annotate_flips.py`: Automatically detects and categorizes flips
- `analyze_results.py`: Generates visualizations and statistics
- `summary_dashboard.png`: Visualization dashboard showing key metrics
- `cot-flip-dataset.zip`: Complete packaged dataset (477 examples with mock data - ready structure for real data)
- `package_dataset.py`: Packages everything for public release
