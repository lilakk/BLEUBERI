from typing import Callable, List, Dict, Any, Tuple
from datasets import DatasetDict, Dataset, load_dataset
import evaluate
import numpy as np
import gc
import time
import torch
import sys
import glob
import os
import ast
from model2vec import StaticModel
from transformers import AutoModel
import requests
import json
import re
from bert_score import BERTScorer

class BaseDataset:
    def __init__(self, path, tokenizer):
        self.path = path
        self.data = None
        self.system_message = None
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def get_dataset(self):
        return self.data

    def get_system_message(self)->(Any):
        return self.system_message

    def save_to_disk(self, disk_path):
        self.data.save_to_disk(disk_path)

    def load_dataset(self, split:str, cache_dir:str="", streaming:bool=False, shuffle:bool=False, from_disk:str=None)->None:
        raise NotImplementedError("To be implemented by child class")


class KeywordDataset(BaseDataset):
    def __init__(self, path, tokenizer):
        super().__init__(path, tokenizer)
        self.system_message = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.bertscore = BERTScorer(model_type="distilbert-base-uncased")

    def load_dataset(self, split:str="train", cache_dir:str="", streaming:bool=False, shuffle:bool=False, load_from_disk:bool=False)->None:
        if load_from_disk:
            self.data = DatasetDict.load_from_disk(self.path)
            self.data = self.data[split]
        else:
            files = glob.glob(os.path.join(self.path, "*"))
            data_file = [f for f in files if split in f][0]  # temporary solution; we're parsing split from file name
            print(f"Loading dataset from {data_file}")
            data = load_dataset("csv", data_files=data_file)['train']  # loading hf dataset from csv will automatically create a train split
            if shuffle:
                data = data.shuffle(seed=42)

            # we want to run ast literal eval because loading from csv will load keyword lists as strings
            def convert_keywords(example):
                try:
                    if isinstance(example['keywords'], str):
                        example['keywords'] = ast.literal_eval(example['keywords'])
                    return example
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing keywords: {e}")
                    return example
            
            data = data.map(convert_keywords)
            self.data = data
    
    def add_conversation_format(self, reasoning=False):
        def _add_template(example):
            assert "messages" in example, "messages must be provided"
            prompt_msg = []
            if reasoning:
                prompt_msg.append(
                    {
                        "role": "system",
                        "content": self.system_message
                    }
                )
            user_msg = next(msg for msg in example['messages'] if msg['role'] == "user")
            prompt_msg.append(user_msg)
            example["prompt"] = prompt_msg
            return example
        self.data = self.data.map(_add_template)
    
    def extract_answer(self, completion):
        if "<answer>" in completion and "</answer>" in completion:
            return completion.split("<answer>")[1].split("</answer>")[0].strip()
        elif "<answer>" in completion:
            return completion.split("<answer>")[1].strip()
        else:
            return completion.strip()

    def bleu_reward_func(self, completions, **kwargs):
        scores = []
        for completion, references in zip(completions, kwargs["references"]):
            if isinstance(completion, list) and "role" in completion[0]:
                completion = next(msg for msg in completion if msg["role"] == "assistant")["content"]
            if isinstance(completion, float) or (isinstance(completion, str) and len(completion.strip()) == 0):
                scores.append(0)
            else:
                completion = self.extract_answer(completion)
                if len(completion.strip()) == 0:
                    scores.append(0)
                else:
                    bleu_score = self.bleu.compute(predictions=[completion], references=[references], smooth=True)
                    scores.append(bleu_score["bleu"])
        return scores
    
    def rouge_reward_func(self, completions, **kwargs):
        scores = []
        for completion, references in zip(completions, kwargs["references"]):
            if isinstance(completion, list) and "role" in completion[0]:
                completion = next(msg for msg in completion if msg["role"] == "assistant")["content"]
            if isinstance(completion, float) or (isinstance(completion, str) and len(completion.strip()) == 0):
                scores.append(0)
            else:
                completion = self.extract_answer(completion)
                rouge_score = self.rouge.compute(predictions=[completion], references=[references])
                scores.append(rouge_score["rougeL"])
        return scores
    
    def bleu_rouge_f1_reward_func(self, completions, **kwargs):
        bleu_scores = self.bleu_reward_func(completions, **kwargs)
        rouge_scores = self.rouge_reward_func(completions, **kwargs)
        combined_scores = [
            2 * bleu * rouge / (bleu + rouge) if (bleu + rouge) > 0 else 0.0
            for bleu, rouge in zip(bleu_scores, rouge_scores)
        ]
        return combined_scores
    
    def bertscore_reward_func(self, completions, **kwargs):
        scores = []
        all_completions = []
        all_references = kwargs["references"]
        for completion in completions:
            if isinstance(completion, list) and "role" in completion[0]:
                completion = next(msg for msg in completion if msg["role"] == "assistant")["content"]
            all_completions.append(completion)
        P, R, F1 = self.bertscore.score(all_completions, all_references)
        scores = F1.tolist()
        return scores
    
    def rm_reward_func(self, completions, **kwargs):
        url = "http://192.222.51.94:8001/score"
        
        batch_conversations = []
        for prompt, completion in zip(kwargs["prompts"], completions):
            if isinstance(completion, list) and "role" in completion[0]:
                completion = next(msg for msg in completion if msg["role"] == "assistant")["content"]
            completion = self.extract_answer(completion)
            completion_formatted = {
                "role": "assistant",
                "content": completion
            }
            conversation = [
                prompt[0],  # prompt is a list containing a dict with role and content for the user message
                completion_formatted
            ]
            batch_conversations.append(conversation)
        
        data = {"batch_conversations": batch_conversations}
        
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            return result["scores"]
        else:
            print(f"Error in rm_reward_func: {response.status_code}")
            print(f"Error response: {response.text}")
            import pdb; pdb.set_trace()
            return [0.0] * len(completions)
    
    def bleu_rm_reward_func(self, completions, **kwargs):
        group_size = len(completions)

        bleu_scores = torch.tensor(self.bleu_reward_func(completions, **kwargs), dtype=torch.float32)
        rm_scores = torch.tensor(self.rm_reward_func(completions, **kwargs), dtype=torch.float32)

        def zscore_groupwise(tensor, group_size, eps=1e-6):
            reshaped = tensor.view(-1, group_size)
            mean = reshaped.mean(dim=1, keepdim=True)
            std = reshaped.std(dim=1, keepdim=True)
            return ((reshaped - mean) / (std + eps)).view(-1)

        bleu_normed = zscore_groupwise(bleu_scores, group_size)
        rm_normed = zscore_groupwise(rm_scores, group_size)

        final_scores = ((bleu_normed + rm_normed) / 2).tolist()
        return final_scores
    
    def format_reward_func(self, completions, **kwargs):
        pattern = r"^<think>.*?</think><answer>.*?</answer>$"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    def get_reward_funcs(self, reward_func_names=["bleu"]):
        reward_funcs = []
        for func_name in reward_func_names:
            if func_name == "bleu":
                reward_funcs.append(self.bleu_reward_func)
            elif func_name == "bertscore":
                reward_funcs.append(self.bertscore_reward_func)
            elif func_name == "rm":
                reward_funcs.append(self.rm_reward_func)
            elif func_name == "format":
                reward_funcs.append(self.format_reward_func)
            elif func_name == "rouge":
                reward_funcs.append(self.rouge_reward_func)
            elif func_name == "bleu_rouge_f1":
                reward_funcs.append(self.bleu_rouge_f1_reward_func)
            elif func_name == "bleu_rm":
                reward_funcs.append(self.bleu_rm_reward_func)
        
        if not reward_funcs:
            print("No valid reward functions specified, using bleu as default")
            reward_funcs = [self.bleu_reward_func]
            
        return reward_funcs
