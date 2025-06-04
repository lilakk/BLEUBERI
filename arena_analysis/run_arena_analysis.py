from datasets import load_dataset, load_from_disk
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import torch
import evaluate
import random
import os
import pandas as pd
import numpy as np

random.seed(42)

def bleu_reward(prediction, references):
    bleu_score = bleu.compute(predictions=[prediction], references=[references], smooth=True)
    return bleu_score["bleu"]

def bleu_no_bp(prediction, references):
    bleu_score = bleu.compute(predictions=[prediction], references=[references], smooth=True)
    return bleu_score["bleu"] / bleu_score["brevity_penalty"]

def bleu_bp_only(prediction, references):
    bleu_score = bleu.compute(predictions=[prediction], references=[references], smooth=True)
    return bleu_score["brevity_penalty"]

def rouge_reward(prediction, references):
    rouge_score = rouge.compute(predictions=[prediction], references=[references])
    return rouge_score["rougeL"]

def bleu_rouge_f1_reward(prediction, references):
    bleu_score = bleu.compute(predictions=[prediction], references=[references], smooth=True)
    rouge_score = rouge.compute(predictions=[prediction], references=[references])
    return (bleu_score["bleu"] * rouge_score["rougeL"] * 2) / (bleu_score["bleu"] + rouge_score["rougeL"] + 1e-10)

def bertscore_reward(prediction, references):
    bertscore_score = bertscore.compute(predictions=[prediction], references=[references], model_type="distilbert-base-uncased")
    return bertscore_score["f1"][0]

def random_score_func(prediction, references):
    return random.random()

def longer_score_func(prediction, references):
    return len(prediction)

def evaluate_instruction_following(instruction: str, response: str, rm_model, rm_tokenizer) -> float:
    messages = [{'role': "user", "content": instruction}, {'role': "assistant", "content": response}]
    tokenized_message = rm_tokenizer.apply_chat_template(
        messages, 
        tokenize=True,
        return_tensors="pt"
    ).cuda()
    
    with torch.no_grad():
        score = rm_model(tokenized_message).logits[0][0].item()
    return score

METRICS = [
    {
        "name": "bleu",
        "function": bleu_reward,
        "score_a_col": "bleu_score_a",
        "score_b_col": "bleu_score_b",
        "winner_col": "bleu_winner",
        "alignment_col": "bleu_alignment"
    },
    {
        "name": "bleu_no_bp",
        "function": bleu_no_bp,
        "score_a_col": "bleu_no_bp_score_a",
        "score_b_col": "bleu_no_bp_score_b",
        "winner_col": "bleu_no_bp_winner",
        "alignment_col": "bleu_no_bp_alignment"
    },
    {
        "name": "bleu_bp_only",
        "function": bleu_bp_only,
        "score_a_col": "bleu_bp_only_score_a",
        "score_b_col": "bleu_bp_only_score_b",
        "winner_col": "bleu_bp_only_winner",
        "alignment_col": "bleu_bp_only_alignment"
    },
    {
        "name": "rouge",
        "function": rouge_reward,
        "score_a_col": "rouge_score_a",
        "score_b_col": "rouge_score_b",
        "winner_col": "rouge_winner",
        "alignment_col": "rouge_alignment"
    },
    {
        "name": "bertscore",
        "function": bertscore_reward,
        "score_a_col": "bertscore_score_a",
        "score_b_col": "bertscore_score_b",
        "winner_col": "bertscore_winner",
        "alignment_col": "bertscore_alignment"
    },
    {
        "name": "bleu_rouge_f1",
        "function": bleu_rouge_f1_reward,
        "score_a_col": "bleu_rouge_f1_score_a",
        "score_b_col": "bleu_rouge_f1_score_b",
        "winner_col": "bleu_rouge_f1_winner",
        "alignment_col": "bleu_rouge_f1_alignment"
    },
    {
        "name": "random",
        "function": random_score_func,
        "score_a_col": "random_score_a",
        "score_b_col": "random_score_b",
        "winner_col": "random_winner",
        "alignment_col": "random_alignment"
    },
    {
        "name": "longer",
        "function": longer_score_func,
        "score_a_col": "longer_score_a",
        "score_b_col": "longer_score_b",
        "winner_col": "longer_winner",
        "alignment_col": "longer_alignment"
    }
]

RM_METRICS = [
    {
        "name": "rm_8b",
        "model_name": "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
        "score_a_col": "rm_8b_score_a",
        "score_b_col": "rm_8b_score_b",
        "winner_col": "rm_8b_winner",
        "alignment_col": "rm_8b_alignment"
    },
    {
        "name": "rm_27b",
        "model_name": "Skywork/Skywork-Reward-Gemma-2-27B-v0.2",
        "score_a_col": "rm_27b_score_a",
        "score_b_col": "rm_27b_score_b",
        "winner_col": "rm_27b_winner",
        "alignment_col": "rm_27b_alignment"
    }
]

ds_path = "arena_1k_final"
ds = load_from_disk(ds_path)

ids = [s["id"] for s in ds]
all_ref_models = [
    "gemini-2.5-pro-exp-03-25",
    "deepseek-chat-v3-0324",
    "o4-mini-2025-04-16",
    "claude-3-7-sonnet@20250219",
    "qwen-max",
    "gpt-4o-2024-08-06",
    "Meta-Llama-3-8B-Instruct",
    "OLMo-2-1124-7B-Instruct",
    "Qwen2.5-0.5B-Instruct",
    "Llama-3.1-8B-Instruct"
]
ref_model_configs = [
    ["gemini-2.5-pro-exp-03-25"],
    ["deepseek-chat-v3-0324"],
    ["o4-mini-2025-04-16"],
    ["claude-3-7-sonnet@20250219"],
    ["qwen-max"],
    ["gpt-4o-2024-08-06"],
    ["Meta-Llama-3-8B-Instruct"],
    ["OLMo-2-1124-7B-Instruct"],
    ["Qwen2.5-0.5B-Instruct"],
    ["Llama-3.1-8B-Instruct"],
    ["gemini-2.5-pro-exp-03-25", "deepseek-chat-v3-0324"],
    ["gemini-2.5-pro-exp-03-25", "deepseek-chat-v3-0324", "o4-mini-2025-04-16"],
    ["gemini-2.5-pro-exp-03-25", "deepseek-chat-v3-0324", "o4-mini-2025-04-16", "claude-3-7-sonnet@20250219"],
    ["gemini-2.5-pro-exp-03-25", "deepseek-chat-v3-0324", "o4-mini-2025-04-16", "claude-3-7-sonnet@20250219", "qwen-max"]
]

model_outputs = {}
for model in all_ref_models:
    model_file = f"model_outputs/{model}.csv"
    df = pd.read_csv(model_file)
    df = df[df['id'].isin(ids)]
    if len(df) < len(ds):
        print(f"Warning: {model} has only {len(df)} outputs for {len(ds)} records")
    outputs = {}
    for idx, row in df.iterrows():
        record_id = row['id']
        outputs[record_id] = row['output']
    model_outputs[model] = outputs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# for any config, we only want the common IDs shared by all ref models
common_ids = set(model_outputs[all_ref_models[0]].keys())
for model in all_ref_models[1:]:
    common_ids &= set(model_outputs[model].keys())

# filter out IDs where any model's response is NaN
ids_with_responses = set()
for ex_id in common_ids:
    all_valid = True
    for model in all_ref_models:
        if pd.isna(model_outputs[model][ex_id]):
            all_valid = False
            break
    if all_valid:
        ids_with_responses.add(ex_id)

valid_ids = set()
for ex in ds:
    if ex["id"] not in ids_with_responses:
        continue
    if ex["winner"] == "tie (bothbad)":
        continue
    valid_ids.add(ex["id"])
print(f"# of valid ids: {len(valid_ids)}")

rm_results_file = "arena_1k_final_results/rm_judgments_all.csv"
if not os.path.exists(rm_results_file):
    print("Running RM judgments with both 8B and 27B models...")
    
    rm_models = {}
    rm_tokenizers = {}
    
    # for rm_config in RM_METRICS:
    #     print(f"Loading {rm_config['name']} model: {rm_config['model_name']}")
    #     rm_models[rm_config['name']] = AutoModelForSequenceClassification.from_pretrained(
    #         rm_config['model_name'],
    #         torch_dtype=torch.bfloat16,
    #         device_map="auto",
    #         attn_implementation="flash_attention_2",
    #         num_labels=1
    #     )
    #     rm_models[rm_config['name']].eval()
    #     rm_tokenizers[rm_config['name']] = AutoTokenizer.from_pretrained(rm_config['model_name'])
    
    rm_results = []
    for dd in tqdm(ds):
        if dd["id"] not in valid_ids:
            continue
        
        instruction = next((p["content"] for p in dd["conversation_a"] if p["role"] == "user"), None)
        response_a = next((p["content"] for p in dd["conversation_a"] if p["role"] == "assistant"), None)
        response_b = next((p["content"] for p in dd["conversation_b"] if p["role"] == "assistant"), None)
        
        if response_a is None or response_b is None:
            continue
        
        result = {
            "id": dd["id"],
        }
        
        for rm_config in RM_METRICS:
            rm_name = rm_config['name']
            rm_model = rm_models[rm_name]
            rm_tokenizer = rm_tokenizers[rm_name]
            
            rm_score_a = evaluate_instruction_following(instruction, response_a, rm_model, rm_tokenizer)
            rm_score_b = evaluate_instruction_following(instruction, response_b, rm_model, rm_tokenizer)
            rm_winner = "model_a" if rm_score_a > rm_score_b else "model_b" if rm_score_b > rm_score_a else "tie"
            
            result[rm_config["score_a_col"]] = rm_score_a
            result[rm_config["score_b_col"]] = rm_score_b
            result[rm_config["winner_col"]] = rm_winner
        
        rm_results.append(result)
    
    rm_df = pd.DataFrame(rm_results)
    rm_df.to_csv(rm_results_file, index=False)
    print(f"Saved RM judgments to {rm_results_file}")
    
    rm_json_file = "arena_1k_final_results/rm_judgments_all.json"
    rm_df.to_json(rm_json_file, orient='records', indent=4)
    print(f"Saved RM judgments to {rm_json_file}")
    
    for rm_name in list(rm_models.keys()):
        del rm_models[rm_name]
        del rm_tokenizers[rm_name]
    torch.cuda.empty_cache()
else:
    print(f"Loading RM judgments from {rm_results_file}")
    rm_df = pd.read_csv(rm_results_file)

for config_idx, ref_models in enumerate(ref_model_configs):
    config_name = "arena_benchmark_results"
    for model in ref_models:
        config_name += f"_{model.split('-')[0]}"
    result_path = f"arena_1k_final_results/{config_name}.csv"
    
    print(f"\nConfiguration {config_idx+1}: {ref_models}")
    
    if os.path.exists(result_path):
        print(f"Result file {result_path} already exists, skipping...")
        continue
    
    results_df = pd.DataFrame()
    existing_ids = set()
    new_metrics = METRICS
    new_rm_metrics = RM_METRICS
    print(f"Creating new results file {result_path}")
    
    new_results = []
    updated = False
    
    for dd in tqdm(ds):
        if dd["id"] not in valid_ids:
            continue
        
        instruction = next((p["content"] for p in dd["conversation_a"] if p["role"] == "user"), None)
        response_a = next((p["content"] for p in dd["conversation_a"] if p["role"] == "assistant"), None)
        response_b = next((p["content"] for p in dd["conversation_b"] if p["role"] == "assistant"), None)
        
        if response_a is None or response_b is None:
            continue
            
        references = [model_outputs[model][dd["id"]] for model in ref_models]
        
        if dd["id"] in existing_ids:
            existing_row = results_df[results_df['id'] == dd["id"]].iloc[0].to_dict()
            
            missing_metrics = []
            for metric in METRICS:
                if (metric["score_a_col"] not in existing_row or 
                    pd.isna(existing_row[metric["score_a_col"]]) or
                    metric["score_b_col"] not in existing_row or 
                    pd.isna(existing_row[metric["score_b_col"]]) or
                    metric in new_metrics):
                    missing_metrics.append(metric)
            
            missing_rm_metrics = []
            for rm_metric in RM_METRICS:
                if (rm_metric["score_a_col"] not in existing_row or 
                    pd.isna(existing_row[rm_metric["score_a_col"]]) or
                    rm_metric["score_b_col"] not in existing_row or 
                    pd.isna(existing_row[rm_metric["score_b_col"]]) or
                    rm_metric in new_rm_metrics):
                    missing_rm_metrics.append(rm_metric)
            
            if not missing_metrics and not missing_rm_metrics:
                continue
                
            result = existing_row.copy()
            for metric in missing_metrics:
                score_a = metric["function"](response_a, references)
                score_b = metric["function"](response_b, references)
                winner = "model_a" if score_a > score_b else "model_b" if score_b > score_a else "tie"
                
                result[metric["score_a_col"]] = score_a
                result[metric["score_b_col"]] = score_b
                result[metric["winner_col"]] = winner
                result[metric["alignment_col"]] = 1 if winner == dd["winner"] else 0
            
            for rm_metric in missing_rm_metrics:
                rm_row = rm_df[rm_df['id'] == dd["id"]]
                if len(rm_row) > 0:
                    rm_score_a = rm_row[rm_metric["score_a_col"]].values[0]
                    rm_score_b = rm_row[rm_metric["score_b_col"]].values[0]
                    rm_winner = rm_row[rm_metric["winner_col"]].values[0]
                    
                    result[rm_metric["score_a_col"]] = rm_score_a
                    result[rm_metric["score_b_col"]] = rm_score_b
                    result[rm_metric["winner_col"]] = rm_winner
                    result[rm_metric["alignment_col"]] = 1 if rm_winner == dd["winner"] else 0
            
            for key, value in result.items():
                results_df.loc[results_df['id'] == dd["id"], key] = value
            
            updated = True
            
        else:
            rm_row = rm_df[rm_df['id'] == dd["id"]]
            if len(rm_row) == 0:
                continue
            
            result = {
                "id": dd["id"],
                "instruction": instruction,
                "response_a": response_a,
                "response_b": response_b,
                "winner": dd["winner"],
                "model_a": dd["model_a"],
                "model_b": dd["model_b"],
                "judge": dd["judge"],
            }
            
            for rm_metric in RM_METRICS:
                rm_score_a = rm_row[rm_metric["score_a_col"]].values[0]
                rm_score_b = rm_row[rm_metric["score_b_col"]].values[0]
                rm_winner = rm_row[rm_metric["winner_col"]].values[0]
                
                result[rm_metric["score_a_col"]] = rm_score_a
                result[rm_metric["score_b_col"]] = rm_score_b
                result[rm_metric["winner_col"]] = rm_winner
                result[rm_metric["alignment_col"]] = 1 if rm_winner == dd["winner"] else 0
            
            for metric in METRICS:
                score_a = metric["function"](response_a, references)
                score_b = metric["function"](response_b, references)
                winner = "model_a" if score_a > score_b else "model_b" if score_b > score_a else "tie"
                
                result[metric["score_a_col"]] = score_a
                result[metric["score_b_col"]] = score_b
                result[metric["winner_col"]] = winner
                result[metric["alignment_col"]] = 1 if winner == dd["winner"] else 0
            
            new_results.append(result)
    
    if new_results:
        new_df = pd.DataFrame(new_results)
        results_df = pd.concat([results_df, new_df], ignore_index=True)
        updated = True
    
    if updated or not os.path.exists(result_path):
        results_df.to_csv(result_path, index=False)
        print(f"Saved/updated results to {result_path}")
        
        json_path = f"arena_1k_final_results/{config_name}.json"
        results_df.to_json(json_path, orient='records', indent=4)
        print(f"Saved/updated JSON results to {json_path}")
    else:
        print("No changes made to existing results")
    
    print("\nAlignment scores:")
    for metric in METRICS:
        alignment_score = results_df[metric["alignment_col"]].mean()
        print(f"{metric['name'].upper()}: {alignment_score:.4f}")
    
    for rm_metric in RM_METRICS:
        alignment_score = results_df[rm_metric["alignment_col"]].mean()
        print(f"{rm_metric['name'].upper()}: {alignment_score:.4f}")

aggregate_results_path = "arena_1k_final_results/arena_1k_new_filtered_aggregate.json"
config_results = []

for ref_models in ref_model_configs:
    config_name = "arena_benchmark_results"
    for model in ref_models:
        config_name += f"_{model.split('-')[0]}"
    result_path = f"arena_1k_final_results/{config_name}.csv"
    
    if not os.path.exists(result_path):
        print(f"Results file {result_path} doesn't exist, skipping in aggregate...")
        continue
    
    results_df = pd.read_csv(result_path)
    
    row = {
        "config_name": config_name,
        "ref_models": ", ".join(ref_models),
        "num_ref_models": len(ref_models)
    }
    
    for metric in METRICS:
        if metric["alignment_col"] in results_df.columns:
            alignment_score = results_df[metric["alignment_col"]].mean()
            row[f"{metric['name']}_alignment"] = alignment_score
    
    for rm_metric in RM_METRICS:
        if rm_metric["alignment_col"] in results_df.columns:
            alignment_score = results_df[rm_metric["alignment_col"]].mean()
            row[f"{rm_metric['name']}_alignment"] = alignment_score
    
    config_results.append(row)

if config_results:
    aggregate_df = pd.DataFrame(config_results)
    cols = ["config_name", "ref_models", "num_ref_models"]
    
    metric_cols = []
    for metric in METRICS:
        metric_cols.append(f"{metric['name']}_alignment")
    for rm_metric in RM_METRICS:
        metric_cols.append(f"{rm_metric['name']}_alignment")
    
    metric_cols = [col for col in metric_cols if col in aggregate_df.columns]
    cols.extend(metric_cols)
    
    aggregate_df = aggregate_df[cols]
    aggregate_df.to_json(aggregate_results_path, orient='records', indent=4)
    print(f"Saved aggregate JSON results to {aggregate_results_path}")
else:
    print("\nNo configurations found with results, skipping aggregate file creation.")
