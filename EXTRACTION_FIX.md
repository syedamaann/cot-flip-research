# Answer Extraction Fix

## Problem
The original `extract_cot_and_answer()` function failed to extract final answers correctly from DeepSeek-R1 model outputs, resulting in ~80-90% extraction failures.

### Common Issues
- Extracted `"**"` instead of actual answers from `\boxed{}` format
- Extracted punctuation marks like `"."` instead of answers
- Failed to handle `</think>` tag format used by DeepSeek-R1
- Extracted random sentence fragments instead of final answers

## Solution
Implemented priority-based extraction with proper DeepSeek-R1 support:

1. **LaTeX `\boxed{}` format** (Priority 1) - Most reliable for math answers
2. **`</think>` tag format** (Priority 2) - DeepSeek-R1's reasoning format
3. **Standard answer markers** (Priority 3) - "Answer:", "Final Answer:", etc.
4. **Last line fallback** (Priority 4) - When no markers found

### Key Improvements
- Finds LAST `\boxed{}` occurrence for multi-step problems
- Properly separates reasoning from answer in `</think>` format
- Cleans up markdown artifacts (`**`, standalone dots)
- More robust CoT/answer boundary detection

## Generating Fresh Data

### Quick Test (50 problems)
```bash
python3 collect_data.py \
  --backend vllm \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --max 50 \
  --batch-size 16 \
  --checkpoint-every 10 \
  --output ./data/raw/test_outputs.json
```

### Full Dataset (493 problems)
```bash
python3 collect_data.py \
  --backend vllm \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --batch-size 16 \
  --checkpoint-every 50 \
  --output ./data/raw/model_outputs.json
```

Expected time: ~25-30 minutes with batch size 16

## Validation
After generation, check for suspicious extractions:
```python
import json

with open('data/raw/model_outputs.json') as f:
    data = json.load(f)

suspicious = [
    entry for entry in data
    if len(entry['final_answer']) < 2 or entry['final_answer'] in ['**', '.', '']
]

print(f"Suspicious extractions: {len(suspicious)} / {len(data)}")
for entry in suspicious[:5]:
    print(f"  {entry['problem_id']}: '{entry['final_answer']}'")
```

## Files Changed
- `collect_data.py` - Improved `extract_cot_and_answer()` function
- `reextract_answers.py` - Utility to re-extract from existing data (not needed for fresh generation)
- `data/raw/.gitignore` - Prevent committing large generated files
