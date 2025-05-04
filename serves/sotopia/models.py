import os
import re
import json
import time
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from jinja2 import Environment, FileSystemLoader

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

class ToMSampler:
    def __init__(
        self,
        model_path: str,
        base_model: str,
        template_path: str,
        log_path: str = "raw_outputs.jsonl",
        device: str | torch.device | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True) if os.path.dirname(self.log_path) else None

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        adapter_dir = os.path.join(model_path, "adapter_model")
        if os.path.exists(adapter_dir + ".safetensors") or os.path.exists(adapter_dir + ".bin"):
            self.model = PeftModel.from_pretrained(base, model_path)
        else:
            self.model = base
        self.model.eval().to(self.device)

        env = Environment(
            loader=FileSystemLoader(os.path.dirname(template_path))
        )
        self.template = env.get_template(os.path.basename(template_path))

    def format_prompt(self, messages) -> str:
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
        return self.template.render(messages=messages)

    def inference(self, messages, temperature, top_p, max_new_tokens) -> str:
        print("Starting inference...")
        prompt = self.format_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        for attempt in range(3):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                )
            raw = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
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
