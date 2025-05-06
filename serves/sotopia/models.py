import os

import numpy as np
import requests
import torch
from jinja2 import Environment, FileSystemLoader
from peft import PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import json
import time

SYSTEM_PROMPT ="""
You are a helpful AI social assistant, designed to provide well-reasoned, structured, and detailed responses.
You FIRST think about the reasoning process as an internal monologue. Write your reasoning inside <think>...</think> tags. Then provide your final response inside <answer>...</answer> tags.
The content inside <answer> must be a JSON object that strictly conforms to the output schema provided in the user prompt. 
Please provide your answer in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""

class RejectionSampler:
    def __init__(
        self,
        base_model_path,
        grpo_model_path,
        template_path,
        log_path,
    ):
        self.base_model_path = base_model_path
        self.grpo_model_path = grpo_model_path
        self.template_path = template_path

        self.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.grpo_model = self.load_grpo_model(grpo_model_path)
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True) if os.path.dirname(self.log_path) else None

        env = Environment(loader=FileSystemLoader("/".join(template_path.split("/")[:-1])))
        self.template = env.get_template(template_path.split("/")[-1])

    def load_grpo_model(self, grpo_model_path):
        """Load grpo model with optional QLoRA quantization."""
        print(f"Loading reward model: {grpo_model_path}")

        base_model = AutoModelForCausalLM.from_pretrained(self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto")
        base_model.config.pad_token_id = self.tokenizer.pad_token_id

        # Check and load the adapter if it exists
        adapter_path = os.path.join(grpo_model_path, 'adapter_model')
        if os.path.exists(adapter_path + '.safetensors') or os.path.exists(adapter_path + '.bin'):
            print(f"Loading reward adapter from: {grpo_model_path}")
            self.grpo_model = PeftModelForCausalLM.from_pretrained(base_model, grpo_model_path)
        else:
            print(f"No adapter found at {adapter_path}, using base model for reward")
            self.grpo_model = base_model

        self.grpo_model.eval()  # Set to evaluation mode
        return self.grpo_model.to(self.model_device)

    def format_prompt(self, messages, add_generation_prompt=True):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        user_suffix = """
\n\nPlease only generate a response in the following format:
<think>
[your internal reasoning about what to do and why, in natural language]
</think>
<answer>
[a JSON object matching the output schema above]
</answer>
Do not include any other text outside the <think> and <answer> tags. Follow this format strictly.
"""
        for msg in messages:
            if msg["role"] == "user":
                msg["content"] += user_suffix

        if add_generation_prompt is True:
            return self.template.render(
                messages=messages,
                add_generation_prompt=add_generation_prompt,
            )
        elif add_generation_prompt is False:
            return self.template.render(
                messages=messages,
                add_generation_prompt=add_generation_prompt,
            ).strip()

    def inference(self, messages, temperature, top_p, max_new_tokens):
        print("Starting inference...")
        prompt = self.format_prompt(messages, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model_device)

        for attempt in range(3):
            with torch.no_grad():
                output = self.grpo_model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    max_length=4096,
                    do_sample=False,
                    temperature=temperature,
                    top_p=top_p,
                )
            raw = self.tokenizer.decode(output[0], skip_special_tokens=False)
            print(f"[Attempt {attempt+1}] Raw model output:\n{raw}")

            match = re.search(r"<answer>\s*(\{.*?\})\s*</answer>", raw, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                parsed = json.loads(json_str)

                record = {
                    "timestamp": time.time(),
                    "prompt": prompt,
                    "parsed": parsed,
                    "raw_output": raw,
                }
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                return json_str
            else:
                print(f"Attempt {attempt+1}: <answer> block not found. Retrying...")

        raise ValueError("Failed to extract <answer> JSON after 3 attempts.\nLast raw output:\n" + raw)
