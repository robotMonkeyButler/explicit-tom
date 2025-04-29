import json

APPENDED_INSTRUCTION = """
Please only generate a response in the following format:

<think>
[your internal reasoning about what to do and why, in natural language]
</think>
<answer>
[a JSON object matching the output schema above]
</answer>

Do not include any other text outside the <think> and <answer> tags. Follow this format strictly.
"""

def process_dataset(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    for sample in dataset:
        if "input" in sample:
            sample["input"] = sample["input"].rstrip() + "\n\n" + APPENDED_INSTRUCTION.strip()

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(dataset)} samples and saved to {output_path}")

process_dataset("sotopia_pi_round1_qwen_sft_all_with_instruct_string.json", "sotopia_pi_tom_with_r1_prompt.json")