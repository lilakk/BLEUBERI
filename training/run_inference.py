import torch
import argparse
import logging
import os
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from chat_templates import QWEN_CHAT_TEMPLATE, LLAMA_CHAT_TEMPLATE, OLMO_CHAT_TEMPLATE

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

def main():
    from vllm import LLM, SamplingParams
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

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

    parser = argparse.ArgumentParser(description='Model inference script')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data_path', type=str, default='../data/tulu3_50k', help='Path to the data file')
    parser.add_argument('--split', type=str, default=None, help='Split to use for inference')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--out_dir', type=str, default='../data/inference_outputs', help='Output directory')
    args = parser.parse_args()

    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO)
    logging.info("Loading model with vLLM...")

    ds = load_from_disk(args.data_path)
    if args.split:
        ds = ds[args.split]
    logging.info(f"Dataset loaded successfully from {args.data_path}")

    llm, tokenizer = setup_vllm(args)
    logging.info(f"Model loaded successfully with vLLM")

    if "ckpts" in args.model:
        model_basename = args.model.split("/")[-2:]
        model_basename = "_".join(model_basename)
    else:
        model_basename = os.path.basename(args.model)
    save_path = os.path.join(args.out_dir, f"{model_basename}.csv")
    logging.info(f"Saving results to {save_path}")
    os.makedirs(args.out_dir, exist_ok=True)

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
        prompts = []
        for ex in examples_to_process:
            prompt = next((item['content'] for item in ex["messages"] if item['role'] == 'user'), None)
            prompts.append(format_message(prompt))
        
        logging.info(f"Generating responses for {len(prompts)} prompts")
        responses = generate_responses(llm, prompts)
        logging.info(f"Generated {len(responses)} responses")
        
        for j, ex in enumerate(examples_to_process):
            prompt = next((item['content'] for item in ex["messages"] if item['role'] == 'user'), None)
            all_results.append({
                "id": ex["id"],
                "source": ex["source"],
                "prompt": prompt,
                "response": responses[j]
            })
        
        all_results = sorted(all_results, key=lambda x: x["id"])
        pd.DataFrame(all_results).to_csv(save_path, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
        logging.info(f"All results saved to {save_path}")

if __name__ == "__main__":
    main()
