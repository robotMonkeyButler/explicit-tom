import json
from typing import Any, Dict

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

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



class SFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, template, max_length: int, ):
        self.data = self.load_sft_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template

    def load_sft_data(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        rendered_text = self.template.render(
            messages=[
                {"role": "user", "content": item["input"]},
                {"role": "assistant", "content": item["output"]}
            ],
            add_generation_prompt=False
        )

        tokens = self.tokenizer(
            rendered_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        instruction_text = self.template.render(
            messages=[{"role": "user", "content": item["input"]}],
            add_generation_prompt=True, # important
        )
        instruction_tokens = self.tokenizer(
            instruction_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        labels = input_ids.clone()
        instruction_length = instruction_tokens["input_ids"].size(1)
        labels[:, :instruction_length] = -100

        return {
            "input_ids": input_ids.squeeze(),
            "attention_mask": attention_mask.squeeze(),
            "labels": labels.squeeze(),
        }

    def collate_fn(self, batch):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in batch], batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in batch], batch_first=True, padding_value=-100
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_masks,
        }
       
class GRPODataset(Dataset):
    def __init__(self, data_path: str, tokenizer, template, max_length: int):
        self.data = self.load_sft_data(data_path)
        self.tokenizer = tokenizer 
        self.max_length = max_length
        self.template = template

    def load_sft_data(self, file_path: str):
        with open(file_path, "r") as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        messages = []
        messages.append({"role": "system", "content": SYSTEM_PROMPT}) 
        messages.append({"role": "user", "content": item["input"]})

        rendered_prompt = self.template.render(
            messages=messages,
            add_generation_prompt=True
        )

        return {
            "prompt": rendered_prompt,
            "completion": item["output"] 
        }