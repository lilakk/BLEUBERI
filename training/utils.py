import torch
import logging
import os
import psutil
import subprocess
import gc
import json
import csv
from pathlib import Path
from os.path import exists
import os
import random 
import numpy as np
import pickle as pkl
import hashlib
from transformers import LlamaForCausalLM, Gemma2ForCausalLM, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from evaluate import load
from typing import List, Optional, Union, Tuple, Dict, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

bleu = load("bleu")
rouge = load("rouge")
bertscore = load("bertscore")

def check_existence(path, isDir=False): 
    if isDir and not path.endswith("/"):
        path += "/"
    pathExists = exists(path)
    if not pathExists:
        return False
    if not isDir:
        filePath = Path(path)
        if not filePath.is_file():
            return False
    return True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(model_path, cache_dir, access_token=None):
    if "llama" in model_path.lower():
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            cache_dir=cache_dir,
            attn_implementation="flash_attention_2",
        )
    elif "gemma" in model_path.lower():
        model = Gemma2ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            cache_dir=cache_dir,
            attn_implementation="flash_attention_2",
        )
    elif "mistral" in model_path.lower() or "qwen" in model_path.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            cache_dir=cache_dir,
            attn_implementation="flash_attention_2",
            token=access_token
        )
    elif "olmo" in model_path.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            cache_dir=cache_dir,
            attn_implementation="flash_attention_2",
            token=access_token
        )
    else:
        raise ValueError("Unrecognized model: {}".format(model_path))
    return model

def shorten_ref_model_name(model_name):
    model_name = model_name.lower()
    if "o4-mini" in model_name:
        return "o4mini"
    elif "deepseek" in model_name:
        return "deepseek"
    elif "claude" in model_name:
        return "claude"
    elif "gemini" in model_name:
        return "gemini"
    elif "llama" in model_name:
        return "llama"
    else:
        return model_name

def get_reward_model(device="cuda"):
    model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    rm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    return rm, rm_tokenizer

def bleu_reward(prediction, references):
    if isinstance(prediction, float) or (isinstance(prediction, str) and len(prediction.strip()) == 0):
        return 0
    bleu_score = bleu.compute(predictions=[prediction], references=[references], smooth=True)
    return bleu_score["bleu"]

def rouge_reward(prediction, references):
    if isinstance(prediction, float) or (isinstance(prediction, str) and len(prediction.strip()) == 0):
        return 0
    rouge_score = rouge.compute(predictions=[prediction], references=[references])
    return rouge_score["rougeL"]

def bleu_rouge_f1_reward(prediction, references):
    bleu_score = bleu_reward(prediction, references)
    rouge_score = rouge_reward(prediction, references)
    return 2 * bleu_score * rouge_score / (bleu_score + rouge_score) if (bleu_score + rouge_score) > 0 else 0.0

def bertscore_reward(prediction, references):
    if isinstance(prediction, float) or (isinstance(prediction, str) and len(prediction.strip()) == 0):
        return 0
    bertscore_score = bertscore.compute(predictions=[prediction], references=[references], model_type="distilbert-base-uncased")
    return bertscore_score["f1"][0]

def rm_reward(predictions, prompts, rm_model=None, rm_tokenizer=None, device="cuda"):
    single_input = not isinstance(predictions, list)
    if single_input:
        predictions = [predictions]
        prompts = [prompts]

    if rm_model is None or rm_tokenizer is None:
        rm_model, rm_tokenizer = get_reward_model(device)
    
    all_scores = []
    
    for i, (prediction, prompt) in enumerate(tqdm(zip(predictions, prompts), desc="Computing reward scores", total=len(predictions), disable=single_input)):
        if isinstance(prediction, float) or (isinstance(prediction, str) and len(prediction.strip()) == 0):
            all_scores.append(0.0)
            continue
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": prediction}
        ]
        conv_tokenized = rm_tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            score = rm_model(conv_tokenized).logits[0][0].item()
        all_scores.append(float(score))
    
    return all_scores[0] if single_input else all_scores

def get_model_name(name):
    if "ckpts" in name:
        model_basename = name.split("/")[-2:]
        model_basename = "_".join(model_basename)
    else:
        model_basename = os.path.basename(name)
    return model_basename

def get_ref_models_str(ref_models_or_count):
    if isinstance(ref_models_or_count, int):
        nrefs = ref_models_or_count
        ref_models_str = ""
    else:
        nrefs = len(ref_models_or_count)
        ref_models_str = "-" + "-".join([shorten_ref_model_name(m) for m in ref_models_or_count])
    return nrefs, ref_models_str

def build_score_path(base_dir, data_path, metric, model, nrefs, ref_models_str=""):
    data_basename = os.path.basename(data_path)
    return os.path.join(
        base_dir,
        f"{data_basename}_{metric}_{get_model_name(model)}_{nrefs}ref{ref_models_str}.json"
    )

def save_histogram(score_values: List[float], metric_name_for_plot: str, title: str, xlabel: str, fig_path: str, bins: int = 50):
    """Helper function to generate and save a histogram."""
    if not score_values:
        print(f"No score values provided for metric '{metric_name_for_plot}', skipping histogram generation for {fig_path}.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(score_values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved score distribution plot to {fig_path}")

def save_scores(scores, score_path):
    os.makedirs(os.path.dirname(score_path), exist_ok=True)
    
    with open(score_path, 'w') as f:
        json.dump(scores, f)
    
    print(f"Saved scores to {score_path}")
    
    if scores: # Ensure scores is not empty
        metric_name = list(scores.values())[0]['metric']
        score_values = [s["score"] for s in scores.values()]
        fig_path = f"{os.path.splitext(score_path)[0]}_distribution.png"
        save_histogram(score_values, metric_name, f"{metric_name.upper()} Score Distribution", f"{metric_name.upper()} Score", fig_path)
    else:
        print("No scores to save, skipping histogram generation.")
