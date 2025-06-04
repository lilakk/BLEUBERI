"""Generate answers using vLLM for local inference.

Usage:
python gen_answer_local.py --model /path/to/model --bench_name arena-hard
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

from add_markdown_info import count_markdown_elements, remove_pattern
from utils import (
    load_questions,
    make_config,
    reorg_answer_file,
    OPENAI_MODEL_LIST,
    temperature_config,
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

def format_messages(prompts):
    messages = []
    for prompt in prompts:
        messages.append([{"role": "user", "content": prompt}])
    return messages

def save_answers(questions, outputs, model_name, answer_file, num_choices=1):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    
    for i, (question, output) in enumerate(zip(questions, outputs)):
        output_text = output.outputs[0].text.strip()
        
        turns = [{"content": output_text}]
        choices = [{"index": 0, "turns": turns}]
        
        # Create answer object
        ans = {
            "question_id": question["question_id"],
            "answer_id": shortuuid.uuid(),
            "model_id": model_name,
            "choices": choices,
            "tstamp": time.time(),
        }
        
        # Add metadata if single turn/choice
        if num_choices == 1 and len(turns) == 1:
            metadata = {"token_len": len(encoding.encode(output_text, disallowed_special=()))}
            ans["conv_metadata"] = metadata | count_markdown_elements(
                remove_pattern(output_text, re.compile("```([^`]*)```")),
                suffix=""
            )
        
        # Write to file
        with open(answer_file, "a") as fout:
            fout.write(json.dumps(ans) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--bench_name", type=str, default="arena-hard-v0.1", help="Benchmark name")
    parser.add_argument("--num_choices", type=int, default=1, help="Number of generations per question")
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
    
    # Extract all prompts
    all_prompts = []
    for question in questions:
        # We assume single-turn conversations as per your request
        if len(question["turns"]) > 0:
            all_prompts.append(question["turns"][0]["content"])
    
    # Prepare messages for batch processing
    all_messages = format_messages(all_prompts)
    
    # Prepare sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # Prepare output file
    if "ckpts" in args.model:
        model_basename = args.model.split("/")[-2:]
        model_basename = "_".join(model_basename)
    else:
        model_basename = os.path.basename(args.model)
    answer_file = os.path.join("data", args.bench_name, "model_answer", f"{model_basename}.jsonl")
    logging.info(f"Output to {answer_file}")
    
    # Check if file already exists
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    if os.path.exists(answer_file) and not args.force:
        logging.info(f"Output file {answer_file} already exists. Skipping processing. Use --force to overwrite.")
        sys.exit(0)
    elif os.path.exists(answer_file):
        logging.info(f"Output file {answer_file} already exists. Overwriting as requested.")
        os.remove(answer_file)
    
    # Process all prompts at once
    logging.info(f"Processing {len(all_messages)} prompts...")
    outputs = llm.chat(
        all_messages,
        sampling_params,
        add_generation_prompt=True
    )
    
    # Save all results
    save_answers(questions, outputs, model_basename, answer_file, args.num_choices)
    
    # Reorganize the answer file
    reorg_answer_file(answer_file)
    logging.info("Done!")