import re
import math
from typing import Callable, List

def strict_format_reward_func(completions, **kwargs) -> List[float]:
    assert isinstance(completions[0], str), f"Expected str but got {type(completions[0])}"
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    matches = [re.match(pattern, completion, flags=re.DOTALL) for completion in completions]
    return [10.0 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> List[float]:
    assert isinstance(completions[0], str), f"Expected str but got {type(completions[0])}"
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    matches = [re.search(pattern, completion, flags=re.DOTALL) for completion in completions]
    return [10.0 if match else 0.0 for match in matches]

def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>") == 1:
            count += 2.5
        if text.count("</think>") == 1:
            count += 2.5
        if text.count("<answer>") == 1:
            count += 2.5
        if text.count("</answer>") == 1:
            count += 2.5
        after_answer = text.split("</answer>")[-1].strip()

        if after_answer:
            count -= len(after_answer) * 0.01

        return count

def tag_count_reward(completions, **kwargs) -> List[float]:
    assert isinstance(completions[0], str), f"Expected str but got {type(completions[0])}"
    return [count_tags(completion) for completion in completions]

def extract_answer(text: str) -> str:
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>")[-1].split("</answer>")[0].strip()
    return text.strip()

def check_calling(completions, **kwargs):
    return [100.0 for _ in completions]

def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "format": soft_format_reward_func,
        "tag": tag_count_reward,
        "check_calling": check_calling}
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs