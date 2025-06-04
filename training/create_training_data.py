import argparse
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json
import logging
import csv
from tqdm import tqdm
from typing import List, Optional, Union, Tuple, Dict, Any
from datasets import load_from_disk, Dataset, DatasetDict, load_dataset
from evaluate import load
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from chat_templates import QWEN_CHAT_TEMPLATE, LLAMA_CHAT_TEMPLATE, OLMO_CHAT_TEMPLATE
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METRIC_FUNCTIONS = {
    "bleu": lambda pred, refs, prompt: bleu_reward(pred, refs),
    "rouge": lambda pred, refs, prompt: rouge_reward(pred, refs),
    "bertscore": lambda pred, refs, prompt: bertscore_reward(pred, refs),
    "bleu_rouge_f1": lambda pred, refs, prompt: bleu_rouge_f1_reward(pred, refs),
}

def score_dataset(data, model_outputs_dict, metric):
    score_field = f"{metric}_score"
    scored_data = []
    scores = {}
    
    rm_model, rm_tokenizer = None, None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if metric == "rm":
        rm_model, rm_tokenizer = get_reward_model(device)
    
    valid_examples = []
    valid_responses = []
    valid_prompts_for_rm = [] # Only populated if metric is 'rm'

    for example in data:
        example_id_str = str(example["id"])
        if example_id_str not in model_outputs_dict:
            print(f"Skipping example {example_id_str}: No model output found")
            continue
            
        response = model_outputs_dict[example_id_str]
        
        valid_examples.append(example)
        valid_responses.append(response)
        if metric == "rm":
            prompt = example.get("prompt") 
            if prompt is None:
                print(f"Warning: Skipping example {example_id_str} for RM scoring due to missing 'prompt'.")
            valid_prompts_for_rm.append(prompt) 

    if not valid_examples:
        print("No valid examples found for scoring")
        return scored_data, scores, score_field

    batch_scores = []
    if metric == "rm":
        temp_valid_examples = []
        temp_valid_responses = []
        temp_valid_prompts_for_rm = []
        for ex, resp, prmpt in zip(valid_examples, valid_responses, valid_prompts_for_rm):
            if prmpt is not None:
                temp_valid_examples.append(ex)
                temp_valid_responses.append(resp)
                temp_valid_prompts_for_rm.append(prmpt)
            else:
                print(f"Skipping example {ex['id']} for RM scoring as prompt is missing.")
        
        valid_examples = temp_valid_examples
        valid_responses = temp_valid_responses
        valid_prompts_for_rm = temp_valid_prompts_for_rm

        if valid_examples: # Only run RM if there are examples left
            batch_scores = rm_reward(valid_responses, valid_prompts_for_rm, rm_model=rm_model, rm_tokenizer=rm_tokenizer, device=device)
        else:
            print("No valid examples with prompts for RM scoring.")

    elif metric in METRIC_FUNCTIONS:
        metric_func = METRIC_FUNCTIONS[metric]
        desc = f"Processing {metric.upper()} scores"
        for response, example in tqdm(zip(valid_responses, valid_examples), desc=desc, total=len(valid_responses)):
            references_for_metric = example.get("references")

            if not isinstance(references_for_metric, list) or not references_for_metric or not all(isinstance(r, str) and r for r in references_for_metric):
                print(f"Warning: Invalid or empty references (expected a non-empty list of non-empty strings) found for example {example['id']} for metric {metric}. Skipping score calculation.")
                batch_scores.append(0.0)
                continue

            score_val = metric_func(response, references_for_metric, None) # Pass the list
            batch_scores.append(score_val)
    else:
        print(f"Unsupported metric: {metric}")
        batch_scores = [0.0] * len(valid_examples)

    # Ensure batch_scores has the same length as valid_examples if any specific metric path failed
    if len(batch_scores) != len(valid_examples):
        print(f"Warning: Mismatch in number of scores ({len(batch_scores)}) and examples ({len(valid_examples)}) for metric {metric}. Padding with 0.0.")
        num_missing = len(valid_examples) - len(batch_scores)
        batch_scores.extend([0.0] * num_missing)

    for example, score in zip(valid_examples, batch_scores):
        example[score_field] = score
        scores[str(example["id"])] = { # Ensure ID is string for consistency
            "score": float(score),
            "metric": metric
        }
        scored_data.append(example)
    
    print(f"Scored {len(scored_data)} examples using {metric} metric")
    return scored_data, scores, score_field

def _get_user_prompt_from_messages(messages: List[Dict[str, str]], example_id: Optional[Any] = None) -> Optional[str]:
    """Extracts the user prompt from a list of messages."""
    if not messages:
        if example_id:
            print(f"Warning: Messages list is empty for example {example_id}.")
        return None
    for item in messages:
        if item.get("role") == "user":
            return item.get("content")
    if example_id:
        print(f"Warning: No user prompt found in messages for example {example_id}.")
    return None

def _get_assistant_response_from_messages(messages: List[Dict[str, str]], example_id: Optional[Any] = None) -> Optional[str]:
    """Extracts the assistant response from a list of messages."""
    if not messages:
        if example_id:
            print(f"Warning: Messages list is empty for example {example_id}.")
        return None
    for item in messages:
        if item.get("role") == "assistant":
            return item.get("content")
    if example_id:
        print(f"Warning: No assistant response found in messages for example {example_id}.")
    return None

def _load_or_compute_scores(args: argparse.Namespace, 
                            current_aggregated_data: List[Dict[str, Any]], 
                            model_outputs_dict: Dict[str, str], 
                            score_cache_dir: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Optional[str]]:
    """
    Loads scores from cache if available, otherwise computes them using score_dataset.
    Returns: scored_data_list, scores_dict, score_field_name
    """
    if not current_aggregated_data:
        print("No aggregated data provided to _load_or_compute_scores. Returning empty.")
        return [], {}, None

    nrefs, ref_models_str = get_ref_models_str(args.ref_models)
    potential_score_file = build_score_path(
        score_cache_dir, 
        args.hf_dataset_path, 
        args.metric, 
        args.model, 
        nrefs, 
        ref_models_str
    )

    final_scored_data = []
    final_scores_dict = {}
    final_score_field = None
    needs_recompute = True # Default to recompute

    if os.path.exists(potential_score_file):
        print(f"INFO: Found existing score file at {potential_score_file}")
        try:
            with open(potential_score_file, 'r') as f:
                loaded_scores_cache = json.load(f)
            
            if not loaded_scores_cache or not isinstance(list(loaded_scores_cache.values())[0], dict) or 'metric' not in list(loaded_scores_cache.values())[0] or list(loaded_scores_cache.values())[0]['metric'] != args.metric: # Also check if metric matches
                print(f"Warning: Loaded score file {potential_score_file} seems empty, invalid, or for a different metric. Recomputing scores.")
            else:
                print(f"INFO: Using existing scores from {potential_score_file}")
                final_score_field = f"{args.metric}_score" # Construct based on current args.metric
                final_scores_dict = loaded_scores_cache
                temp_scored_data = []
                missing_count = 0
                for example in current_aggregated_data:
                    example_id_str = str(example["id"])
                    if example_id_str in final_scores_dict:
                        example[final_score_field] = final_scores_dict[example_id_str]["score"]
                        temp_scored_data.append(example)
                    else:
                        print(f"Warning: No score found for example {example_id_str} in {potential_score_file}. Skipping.")
                        missing_count += 1
                final_scored_data = temp_scored_data
                if missing_count > 0:
                    print(f"Warning: Skipped {missing_count} examples due to missing scores in the loaded file.")
                print(f"Applied existing {final_score_field} scores to {len(final_scored_data)} examples")
                needs_recompute = False # Scores loaded successfully
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {potential_score_file}. Recomputing scores.")
        except Exception as e:
            print(f"Warning: Error reading score file {potential_score_file}: {e}. Recomputing scores.")

    if needs_recompute:
        print(f"Scoring {len(current_aggregated_data)} examples with model {args.model} using {args.metric} metric")
        processed_scored_data, computed_scores_dict, computed_score_field = score_dataset(
            current_aggregated_data, model_outputs_dict, args.metric
        )
        final_scored_data = processed_scored_data
        final_scores_dict = computed_scores_dict
        final_score_field = computed_score_field

        # Save the newly computed scores
        os.makedirs(score_cache_dir, exist_ok=True) # Ensure dir exists before saving
        save_scores(final_scores_dict, potential_score_file)
        print(f"Saved computed scores to {potential_score_file}")
        
    return final_scored_data, final_scores_dict, final_score_field

def make_grpo_data(args):
    print("Creating GRPO dataset...")
    print(f"Received ref_models as: {args.ref_models}")

    # Check if the final dataset directory already exists if a specific dataset name is provided
    if args.output_dataset_name:
        grpo_base_output_dir = os.path.join(args.output_dir, "data_grpo")
        potential_output_path = os.path.join(grpo_base_output_dir, args.output_dataset_name)
        if os.path.exists(potential_output_path) and os.path.isdir(potential_output_path):
            print(f"Final dataset directory already exists at {potential_output_path}. Skipping generation.")
            return potential_output_path
    
    # Load dataset once for GRPO process, potentially passing it to run_inference
    print(f"Loading source dataset from HuggingFace: {args.hf_dataset_path} for GRPO aggregation")
    if args.hf_dataset_path.startswith('yapeichang/') or '/' in args.hf_dataset_path:
        source_ds_for_grpo = load_dataset(args.hf_dataset_path, split="train")
    else:
        loaded_ds = load_from_disk(args.hf_dataset_path)
        if isinstance(loaded_ds, DatasetDict):
            source_ds_for_grpo = loaded_ds["train"]
        else:
            source_ds_for_grpo = loaded_ds # Assuming it's already a Dataset object
    print(f"Loaded source dataset with {len(source_ds_for_grpo)} examples for GRPO aggregation.")

    aggregated_data = aggregate_references_for_grpo(args, source_ds_for_grpo) # Pass loaded dataset
    
    scored_data = [] # Initialize with a default
    score_field = None
    scores = {} # To hold scores for saving, if computed
    model_outputs_dict = {}

    if args.selection_mode == "random":
        if not args.num_examples:
            raise ValueError("When selection_mode is 'random', --num_examples must be provided")
        
        scored_data = aggregated_data 
        if len(aggregated_data) > args.num_examples:
            print(f"Randomly sampling {args.num_examples} examples using seed {args.seed}")
            np.random.seed(args.seed) 
            random.seed(args.seed)
            indices = np.random.choice(len(aggregated_data), args.num_examples, replace=False)
            scored_data = [aggregated_data[i] for i in indices]
            print(f"Randomly selected {args.num_examples} examples")
        else:
            print(f"Using all {len(aggregated_data)} examples (requested sample size larger than dataset or equal)")
    else: # Modes: easy, medium, hard (require model outputs and scores)
        if not args.model or not args.metric:
             raise ValueError("For selection_mode 'easy', 'medium', or 'hard', both --model and --metric must be provided.")

        # Step 1: Ensure model inference outputs are available
        inference_base_dir = os.path.join(args.output_dir, "inference_outputs")
        model_basename = get_model_name(args.model)
        inference_results_path = os.path.join(inference_base_dir, f"{model_basename}_inference_results.csv")

        if not os.path.exists(inference_results_path):
            print(f"Inference results not found at {inference_results_path}. Running inference...")
            # Prepare args for run_inference; hf_dataset_path is still needed if source_ds is None
            inference_run_args = argparse.Namespace(
                hf_dataset_path=args.hf_dataset_path, 
                model=args.model,
                output_dir=inference_base_dir,
                seed=args.seed,
                max_new_tokens=args.inference_max_new_tokens
            )
            # Pass the already loaded source_ds_for_grpo to run_inference
            returned_inference_path = run_inference(inference_run_args, preloaded_dataset=source_ds_for_grpo)
            if returned_inference_path is None or not os.path.exists(returned_inference_path):
                raise FileNotFoundError(f"Inference run failed or did not produce the expected output file at {inference_results_path}. Attempted path: {returned_inference_path}")
            inference_results_path = returned_inference_path # Use the path returned by the function
            print(f"Inference complete. Results saved to {inference_results_path}")
        else:
            print(f"Found existing inference results at {inference_results_path}")

        # Load model outputs from the CSV
        print(f"Loading model outputs from {inference_results_path}...")
        df_inference = pd.read_csv(inference_results_path)
        for _, row in df_inference.iterrows():
            if 'id' in row and 'response' in row:
                 model_outputs_dict[str(row['id'])] = row['response'] # Ensure ID is string for consistency
        print(f"Loaded {len(model_outputs_dict)} model outputs.")

        # Filter aggregated_data based on available model outputs
        original_aggregated_count = len(aggregated_data)
        aggregated_data = [ex for ex in aggregated_data if str(ex["id"]) in model_outputs_dict]
        print(f"Filtered aggregated_data from {original_aggregated_count} to {len(aggregated_data)} based on available model outputs.")

        if not aggregated_data:
            print("No examples in aggregated_data after filtering against model outputs. GRPO dataset will be empty or very small.")
            scored_data = [] 
            scores = {}
            score_field = None
        else:
            score_cache_dir = os.path.join(args.output_dir, "scored_outputs")
            scored_data, scores, score_field = _load_or_compute_scores(
                args, 
                aggregated_data, # This is the filtered list
                model_outputs_dict,
                score_cache_dir
            )
        
        # Step 3: Selection logic based on mode and number of examples
        if scored_data and score_field: # Ensure there is data and a score field to sort by
            if args.num_examples and len(scored_data) > args.num_examples:
                if args.selection_mode == "easy":
                    print(f"Selecting the {args.num_examples} highest scoring examples")
                    scored_data = sorted(scored_data, key=lambda x: x[score_field], reverse=True)[:args.num_examples]
                elif args.selection_mode == "medium":
                    print(f"Selecting {args.num_examples} examples from the middle of the distribution")
                    # Sort first to ensure consistent selection
                    temp_sorted_for_medium = sorted(scored_data, key=lambda x: x[score_field])
                    start_idx = (len(temp_sorted_for_medium) - args.num_examples) // 2
                    scored_data = temp_sorted_for_medium[start_idx:start_idx + args.num_examples]
                    # Re-sort by score descending for consistency with other modes if desired, or remove if not needed
                    scored_data = sorted(scored_data, key=lambda x: x[score_field], reverse=True)
                    print(f"Re-sorted selected examples from highest to lowest score")
                else:  # args.selection_mode == "hard"
                    print(f"Selecting the {args.num_examples} lowest scoring examples")
                    scored_data = sorted(scored_data, key=lambda x: x[score_field])[:args.num_examples]
                    # Re-sort by score descending for consistency
                    scored_data = sorted(scored_data, key=lambda x: x[score_field], reverse=True)
                    print(f"Re-sorted selected examples from highest to lowest score")
            else: # Not reducing num_examples or scored_data is already smaller
                scored_data = sorted(scored_data, key=lambda x: x[score_field], reverse=True)
                if args.num_examples is None:
                    print(f"Using all {len(scored_data)} examples, sorted by {score_field.split('_')[0]} score (highest first)")
                # If args.num_examples is set but len(scored_data) <= args.num_examples, all examples are kept and sorted.
            
            print(f"Sorted {len(scored_data)} examples by {score_field.split('_')[0]} score (highest first)")
            if scored_data: 
                 print(f"Highest {score_field} score: {scored_data[0][score_field]}, "
                       f"Lowest {score_field} score: {scored_data[-1][score_field]}")
        elif not scored_data:
            print("No examples left after scoring/filtering to select from.")
        else: # scored_data exists but no score_field (should not happen in this branch)
            print("Warning: Scored data exists but score_field is not set. Cannot sort or select by score.")

    # If selection_mode was "random", scored_data is already set.
    # If other modes, scored_data is now selected and sorted.
    if not scored_data and args.selection_mode != "random":
        print("Warning: No data to save after selection process. Output dataset will be empty.")
        # scored_data could be an empty list here, which is fine for save_grpo_dataset

    output_path = save_grpo_dataset(args, scored_data, score_field)
    return output_path

def aggregate_references_for_grpo(args, source_dataset: Dataset):
    print(f"Aggregating data from reference models: {args.ref_models}")
    os.makedirs(args.output_dir, exist_ok=True)
    ds = source_dataset # Use the passed dataset
    
    print(f"Using pre-loaded dataset with {len(ds)} examples for aggregation.")
    print(f"Dataset columns: {ds.column_names}")
    
    # Map reference model names to dataset column names
    use_gold = "gold" in args.ref_models
    models_to_use = args.ref_models.copy()
    if use_gold:
        models_to_use.remove("gold")
    
    # Create mapping from model names to column names
    model_column_mapping = {}
    available_ref_columns = [col for col in ds.column_names if col.startswith('ref_output_')]
    
    print(f"Available reference output columns: {available_ref_columns}")
    
    for model in models_to_use:
        expected_column = f"ref_output_{model}"
        
        if expected_column in available_ref_columns:
            model_column_mapping[model] = expected_column
            print(f"Mapped model '{model}' to column '{expected_column}'")
        else:
            print(f"Warning: Could not find column '{expected_column}' for model '{model}'")
    
    models_to_use = [model for model in models_to_use if model in model_column_mapping]
    if not models_to_use and not use_gold:
        raise ValueError("No reference models could be mapped to dataset columns")
    
    def has_all_references(example):
        if use_gold and (example.get("ref_output_gold") is None or 
                        pd.isna(example.get("ref_output_gold")) or 
                        str(example.get("ref_output_gold")).strip() == ""):
            return False
        for model in models_to_use:
            col_name = model_column_mapping[model]
            if (example.get(col_name) is None or 
                pd.isna(example.get(col_name)) or 
                str(example.get(col_name)).strip() == ""):
                return False
        return True
    
    ds_filtered = ds.filter(has_all_references)
    
    print(f"After filtering for complete references: {len(ds_filtered)} examples")
    
    aggregated_data = []
    for example in ds_filtered:
        example_id = example.get("id", "unknown_id")
        if "prompt" in example and example["prompt"] is not None:
            prompt = example["prompt"]
        else:
            prompt = _get_user_prompt_from_messages(example.get("messages"), example_id)
        
        if "ref_output_gold" in example and example["ref_output_gold"] is not None:
            ground_truth = example["ref_output_gold"]
        else:
            ground_truth = _get_assistant_response_from_messages(example.get("messages"), example_id)
        
        references = []
        if use_gold:
            references.append(ground_truth)
        for model in models_to_use:
            col_name = model_column_mapping[model]
            references.append(example[col_name])
        
        aggregated_data.append({
            "id": example["id"],
            "source": example.get("source", "unknown"),
            "messages": example["messages"],
            "prompt": prompt,
            "ground_truth": ground_truth,
            "references": references,
        })
    
    print(f"Aggregated {len(aggregated_data)} examples with specified references")
    return aggregated_data

def save_grpo_dataset(args, data, score_field=None):
    data_basename = os.path.basename(args.hf_dataset_path)
    nrefs, ref_models_str = get_ref_models_str(args.ref_models)
    
    model_str = ""
    metric_str = ""
    sampling_str = ""

    if args.selection_mode == "random":
        sampling_str = "_random"
    elif args.selection_mode in ["easy", "medium", "hard"]:
        sampling_str = f"_{args.selection_mode}"

    plot_metric_name = ""
    if args.metric:
        metric_str = f"{args.metric}"
        plot_metric_name = args.metric
        if args.model:
            model_str = f"_{get_model_name(args.model)}_"
    elif score_field:
        plot_metric_name = score_field.replace("_score","")

    grpo_base_output_dir = os.path.join(args.output_dir, "data_grpo")
    
    if args.output_dataset_name:
        dataset_name_suffix = args.output_dataset_name
    else:
        dataset_name_suffix = f"{data_basename}_{metric_str}{model_str}{nrefs}ref{ref_models_str}{sampling_str}_{len(data)}"
    
    output_path = os.path.join(grpo_base_output_dir, dataset_name_suffix)
    
    dataset = DatasetDict({
        "train": Dataset.from_list(data)
    })
    
    if os.path.exists(output_path):
        print(f"Dataset already exists at {output_path}, skipping save.")
    else:
        os.makedirs(output_path, exist_ok=True)
        dataset.save_to_disk(output_path)
        print(f"Saved GRPO dataset with {len(data)} examples to {output_path}")
    
    if score_field and plot_metric_name and data:
        scores_to_plot = [item[score_field] for item in data if score_field in item and pd.notna(item[score_field])]
        if scores_to_plot:
            fig_path = os.path.join(output_path, f"{plot_metric_name}_distribution.png")
            save_histogram(scores_to_plot, plot_metric_name, f"{plot_metric_name.upper()} Score Distribution for GRPO Output", f"{plot_metric_name.upper()} Score", fig_path)
        else:
             print(f"No valid scores found for field '{score_field}' in GRPO output to generate distribution plot.")
    elif not data:
        print("GRPO output data is empty, skipping histogram generation.")

    return output_path

def make_sft_data(args):
    # Construct the potential output path first
    input_basename = os.path.basename(args.input_data_path)
    sft_base_output_dir = os.path.join(args.output_dir, "data_sft")
    dataset_name = f"{input_basename}_SFT"
    output_path = os.path.join(sft_base_output_dir, dataset_name)

    # Check if the dataset already exists
    if os.path.exists(output_path) and os.path.isdir(output_path):
        print(f"SFT dataset already exists at {output_path}. Skipping generation.")
        return output_path

    data = load_from_disk(args.input_data_path)["train"]
    print(f"Processing {len(data)} examples with references...")
    
    def build_message(prompt, response):
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    
    sft_data = []
    for example in tqdm(data, desc="Converting to SFT format"):
        for j, ref in enumerate(example["references"]):
            new_id = f"{example['id']}_{j}"
            sft_data.append({
                "id": new_id,
                "source": example["source"],
                "messages": build_message(example["prompt"], ref),
            })
    
    print(f"Created {len(sft_data)} SFT examples from {len(data)} input examples")
    
    dataset = DatasetDict({
        "train": Dataset.from_list(sft_data),
    })
    os.makedirs(output_path, exist_ok=True) # Ensure dir exists before saving, even if we checked earlier
    dataset.save_to_disk(output_path)
    print(f"Saved SFT dataset with {len(sft_data)} examples to {output_path}")
    
    return output_path

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_message(prompt):
    return [
        {"role": "user", "content": prompt},
    ]

def run_inference(args, preloaded_dataset: Optional[Dataset] = None):
    """Run model inference on the data pool and save results"""
    print("Running model inference...")
    
    try:
        from vllm import LLM, SamplingParams
        from chat_templates import QWEN_CHAT_TEMPLATE, LLAMA_CHAT_TEMPLATE, OLMO_CHAT_TEMPLATE
    except ImportError as e:
        print(f"Error importing vLLM or chat templates: {e}")
        print("Please install vLLM: pip install vllm")
        return None

    def setup_vllm(args):
        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            # hf_token=os.getenv("HF_TOKEN")
        )
        tokenizer = llm.get_tokenizer()
        if not tokenizer.chat_template:
            if "qwen" in args.model.lower():
                tokenizer.chat_template = QWEN_CHAT_TEMPLATE
            elif "llama" in args.model.lower():
                tokenizer.chat_template = LLAMA_CHAT_TEMPLATE
            elif "olmo" in args.model.lower():
                tokenizer.chat_template = OLMO_CHAT_TEMPLATE
            print(f"Chat template set to {tokenizer.chat_template}")
        
        return llm, tokenizer

    def generate_responses(llm, prompts, max_new_tokens=512):
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )
        outputs = llm.chat(
            prompts,
            sampling_params,
            add_generation_prompt=True
        )
        return [output.outputs[0].text.strip() for output in outputs]

    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO)

    if preloaded_dataset is not None:
        print(f"Using pre-loaded dataset for inference with {len(preloaded_dataset)} examples.")
        ds = preloaded_dataset
    else:
        print(f"Loading dataset from HuggingFace: {args.hf_dataset_path} for inference")
        if args.hf_dataset_path.startswith('yapeichang/') or '/' in args.hf_dataset_path:
            ds = load_dataset(args.hf_dataset_path, split="train")
        else:
            loaded_ds = load_from_disk(args.hf_dataset_path)
            if isinstance(loaded_ds, DatasetDict):
                ds = loaded_ds["train"]
            else:
                ds = loaded_ds
        logging.info(f"Dataset loaded successfully from {args.hf_dataset_path} for inference")
        print(f"Loaded dataset with {len(ds)} examples for inference")

    llm, tokenizer = setup_vllm(args)
    logging.info(f"Model loaded successfully with vLLM")

    os.makedirs(args.output_dir, exist_ok=True)
    model_basename = get_model_name(args.model)
    save_path = os.path.join(args.output_dir, f"{model_basename}_inference_results.csv")
    logging.info(f"Saving results to {save_path}")

    # Load existing results if they exist
    existing_results = None
    all_results = []
    if os.path.exists(save_path):
        existing_results = pd.read_csv(save_path)
        all_results = existing_results.to_dict('records')
        logging.info(f"Loaded {len(all_results)} existing results from {save_path}")
        processed_ids = set(existing_results['id'].values)
    else:
        processed_ids = set()

    examples_to_process = [ex for ex in ds if ex['id'] not in processed_ids]
    logging.info(f"Processing {len(examples_to_process)} new examples")
    
    if examples_to_process:
        prompts_for_model = []
        original_prompts_content = []

        for ex in examples_to_process:
            example_id = ex.get("id", "unknown_id")
            if "prompt" in ex and ex["prompt"] is not None:
                current_prompt_content = ex["prompt"]
            else:
                current_prompt_content = _get_user_prompt_from_messages(ex.get("messages"), example_id)
            
            if current_prompt_content is None:
                print(f"Warning: Skipping example {example_id} in inference due to missing prompt.")
                original_prompts_content.append(None)
                prompts_for_model.append(format_message(""))
                continue

            original_prompts_content.append(current_prompt_content)
            prompts_for_model.append(format_message(current_prompt_content))
        
        valid_indices = [i for i, p in enumerate(original_prompts_content) if p is not None]
        prompts_for_model = [prompts_for_model[i] for i in valid_indices]
        examples_to_process_filtered = [examples_to_process[i] for i in valid_indices]
        original_prompts_content_filtered = [original_prompts_content[i] for i in valid_indices]

        if not prompts_for_model:
            print("No valid prompts to send for inference after filtering.")
        else:
            logging.info(f"Generating responses for {len(prompts_for_model)} prompts")
            responses = generate_responses(llm, prompts_for_model, max_new_tokens=args.max_new_tokens)
            logging.info(f"Generated {len(responses)} responses")
            
            response_idx = 0
            for ex, prompt_content in zip(examples_to_process_filtered, original_prompts_content_filtered):
                all_results.append({
                    "id": ex["id"],
                    "source": ex.get("source", "unknown"),
                    "prompt": prompt_content,
                    "response": responses[response_idx]
                })
                response_idx += 1
        
        all_results = sorted(all_results, key=lambda x: x["id"])
        pd.DataFrame(all_results).to_csv(save_path, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
        logging.info(f"All results saved to {save_path}")
        print(f"Inference complete! Results saved to {save_path}")
    else:
        print("All examples have already been processed!")
    
    return save_path

def main():
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser(description="Unified training data generation script")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # GRPO data command
    grpo_parser = subparsers.add_parser("grpo", help="Create GRPO dataset (aggregates references and optionally sorts by score)")
    grpo_parser.add_argument("--hf_dataset_path", type=str, default="yapeichang/BLEUBERI-Tulu3-50k", help="HuggingFace dataset path or local path with reference outputs")
    grpo_parser.add_argument("--inference_max_new_tokens", type=int, default=512, help="Max new tokens for on-the-fly inference if model outputs are missing.")
    grpo_parser.add_argument("--ref_models", type=str, nargs="+", default=["gold"]) # More options: "claude-3-7-sonnet@20250219", "deepseek-chat-v3", "gemini-2.5-pro-exp-03-25", "o4-mini-2025-04-16", "Llama-3.1-8B-Instruct
    grpo_parser.add_argument("--selection_mode", type=str, choices=["random", "easy", "medium", "hard"], default="hard", help="Selection mode: 'random' for random sampling, 'easy' for highest scores, 'medium' for middle scores, 'hard' for lowest scores")
    grpo_parser.add_argument("--output_dataset_name", type=str, default=None, help="Custom name for the GRPO output dataset directory. If None, a name will be automatically generated.")
    grpo_parser.add_argument("--metric", type=str, choices=["bleu", "rm", "rouge", "bertscore", "bleu_rouge_f1"], help="Metric to use for scoring (if not using pre-computed scores)")
    grpo_parser.add_argument("--model", type=str, help="Model to use for scoring (if not using pre-computed scores) and for generating inference outputs if missing.")
    grpo_parser.add_argument("--num_examples", type=int, default=None, help="Number of examples to include in final dataset")
    grpo_parser.add_argument("--output_dir", type=str, default="../data")
    grpo_parser.add_argument("--seed", type=int, default=42, help="Random seed for on-the-fly inference if model outputs are missing.")
    
    # SFT data command
    sft_parser = subparsers.add_parser("sft", help="Convert dataset with references to SFT format")
    sft_parser.add_argument("--input_data_path", type=str, required=True, help="Path to dataset with references")
    sft_parser.add_argument("--output_dir", type=str, default="../data")
    
    args = parser.parse_args()
    
    if args.command == "grpo":
        if args.selection_mode != "random" and not (args.model and args.metric):
            raise ValueError("When selection_mode is 'easy', 'medium', or 'hard', both --model and --metric must be provided")
        make_grpo_data(args)
    elif args.command == "sft":
        make_sft_data(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
