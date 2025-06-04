"""Generate answers using vLLM for local inference for arena-hard-v2.0.

Usage:
python gen_answer_local.py --model /path/to/model --bench_name arena-hard-v2.0
"""
import argparse
import json
import os
import re
import time
import sys
import logging
from tqdm import tqdm

import tiktoken
import shortuuid
import torch
import numpy as np
import random

from utils.add_markdown_info import count_markdown_elements, remove_pattern
from utils.completion import (
    load_questions,
    load_model_answers,
    make_config,
    reorg_answer_file,
)

from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_messages(questions, system_prompt=None):
    messages = []
    for question in questions:
        msg = []
        if system_prompt:
            msg.append({"role": "system", "content": system_prompt})
        msg.append({"role": "user", "content": question["prompt"]})
        messages.append(msg)
    return messages

def extract_answer(completion):
    if "<answer>" in completion and "</answer>" in completion:
        return completion.split("<answer>")[1].split("</answer>")[0].strip()
    elif "<answer>" in completion:
        return completion.split("<answer>")[1].strip()
    else:
        return completion.strip()

def save_answers(questions, outputs, model_name, answer_file):
    encoding = tiktoken.encoding_for_model("gpt-4o")
    
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    
    for i, (question, output) in enumerate(zip(questions, outputs)):
        output_text = output.outputs[0].text.strip()
        
        # Build messages
        messages = []
        if hasattr(output, 'system_prompt') and output.system_prompt:
            messages.append({"role": "system", "content": output.system_prompt})
        messages.append({"role": "user", "content": question["prompt"]})
        assistant_message = {
            "role": "assistant",
            "content": {
                "full_answer": output_text,
                "answer": extract_answer(output_text)
            }
        }
        messages.append(assistant_message)
        
        # Create answer object
        ans = {
            "uid": question["uid"],
            "ans_id": shortuuid.uuid(),
            "model": model_name,
            "messages": messages,
            "tstamp": time.time(),
        }
        
        # Add metadata
        metadata = {
            "token_len": len(encoding.encode(output_text, disallowed_special=()))
        }
        ans["metadata"] = metadata | count_markdown_elements(
            remove_pattern(
                output_text, 
                re.compile("```([^`]*)```")
            ),
            suffix="",
        )
        
        # Write to file
        with open(answer_file, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(ans, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--bench_name", type=str, default="arena-hard-v2.0", help="Benchmark name")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing output file")
    args = parser.parse_args()

    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO)

    from vllm import LLM, SamplingParams
    
    # Initialize vLLM
    logging.info(f"Loading model {args.model}...")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        hf_token=os.getenv("HF_TOKEN")
    )
    
    # Check and set chat template if needed
    tokenizer = llm.get_tokenizer()
    if not tokenizer.chat_template:
        logging.info("Setting chat template to default")
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    
    # Load questions
    question_file = os.path.join("data", args.bench_name, "question.jsonl")
    questions = load_questions(question_file)
    logging.info(f"Loaded {len(questions)} questions from {question_file}")
    
    # Check for existing answers
    if "ckpts" in args.model:
        model_basename = args.model.split("/")[-2:]
        model_basename = "_".join(model_basename)
    else:
        model_basename = os.path.basename(args.model)
    
    answer_file = os.path.join("data", args.bench_name, "model_answer", f"{model_basename}.jsonl")
    existing_answer = {}
    if os.path.exists(answer_file) and not args.force:
        existing_answer = load_model_answers(os.path.join("data", args.bench_name, "model_answer"))
        if model_basename in existing_answer:
            logging.info(f"Found existing answers for {model_basename}")
    
    # Filter out questions that already have answers
    filtered_questions = []
    skipped_count = 0
    for question in questions:
        if model_basename in existing_answer and question["uid"] in existing_answer[model_basename]:
            skipped_count += 1
            continue
        filtered_questions.append(question)
    
    if skipped_count > 0:
        logging.info(f"Skipping {skipped_count} questions that already have answers")
    
    if not filtered_questions:
        logging.info("No new questions to process")
        if not args.force:
            sys.exit(0)
        else:
            filtered_questions = questions
            logging.info("Force flag set, processing all questions")
            os.remove(answer_file)
        
    if "reason" in args.model:
        system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."
    else:
        system_prompt = None
    
    # Prepare messages for batch processing
    all_messages = format_messages(filtered_questions, system_prompt)
    
    # Prepare sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # Process all prompts at once
    logging.info(f"Processing {len(all_messages)} prompts...")
    outputs = llm.chat(
        all_messages,
        sampling_params,
        add_generation_prompt=True
    )
    
    # Save all results
    save_answers(filtered_questions, outputs, model_basename, answer_file)
    
    # Reorganize the answer file
    reorg_answer_file(answer_file)
    logging.info("Done!") 