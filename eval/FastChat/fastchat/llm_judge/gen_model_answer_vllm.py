"""Generate answers with vLLM models.

Usage:
python3 gen_model_answer_vllm.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time

import shortuuid
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import get_conversation_template
# Removed: from fastchat.utils import str_to_torch_dtype
# Removed: import torch

# +++ vLLM Import +++
from vllm import LLM, SamplingParams


def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model, # This will be tensor_parallel_size for vLLM
    num_gpus_total,
    gpu_memory_utilization, # Added for vLLM
    dtype,
    revision,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    num_workers = num_gpus_total // num_gpus_per_model
    use_ray = num_workers > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)( # num_gpus for ray worker, vLLM itself will use tensor_parallel_size
            get_model_answers_vllm
        ).remote
    else:
        get_answers_func = get_model_answers_vllm

    chunk_size = len(questions) // num_workers
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model, # tensor_parallel_size for LLM
                gpu_memory_utilization,
                dtype=dtype,
                revision=revision,
            )
        )

    if use_ray:
        ray.get(ans_handles)


# Removed @torch.inference_mode() as vLLM handles this
def get_model_answers_vllm(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    tensor_parallel_size, # Renamed from num_gpus_per_model for clarity
    gpu_memory_utilization,
    dtype,
    revision,
):
    # Initialize vLLM engine
    llm = LLM(
        model=model_path,
        revision=revision,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype if dtype else "auto", # vLLM uses "auto" or specific strings like "bfloat16"
        trust_remote_code=True, # Often needed for Hugging Face models
    )
    tokenizer = llm.get_tokenizer()

    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        # Adjust temperature for vLLM (0 often means greedy)
        actual_temperature = 0.0 if temperature < 1e-4 else temperature
        
        choices = []
        for i in range(num_choices):
            # Note: vLLM's SamplingParams has a seed, but to match original behavior
            # of reseeding for each choice, we generate one by one if num_choices > 1.
            # If all choices can use the same seed path or rely on vLLM's 'n' param's stochasticity,
            # this loop could be replaced by setting n=num_choices in SamplingParams
            # and processing multiple outputs from a single llm.generate call.
            # For now, keeping the loop to ensure distinct (if stochastic) generation paths per choice.
            # The original script sets torch.manual_seed(i). vLLM's seed is global for the SamplingParams.
            # To get truly different paths for each choice like original, separate generate calls are simpler.

            conv = get_conversation_template(model_id)
            turns = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                # Prepare SamplingParams
                # Ensure stop sequences are lists of strings
                stop_sequences = []
                if conv.stop_str:
                    if isinstance(conv.stop_str, list):
                        stop_sequences.extend(conv.stop_str)
                    else:
                        stop_sequences.append(conv.stop_str)
                
                sampling_params = SamplingParams(
                    n=1, # Generate one completion per call in this loop structure
                    temperature=actual_temperature,
                    max_tokens=max_new_token,
                    stop=stop_sequences if stop_sequences else None,
                    stop_token_ids=conv.stop_token_ids if conv.stop_token_ids else None,
                )

                # some models may error out when generating long outputs
                try:
                    # vLLM generate call
                    request_outputs = llm.generate([prompt], sampling_params)
                    output = request_outputs[0].outputs[0].text

                    # vLLM's output is already decoded text and handles stop conditions.
                    # Additional cleaning for special tokens might still be needed.
                    if tokenizer.special_tokens_map:
                        for special_token_value in tokenizer.special_tokens_map.values():
                            if isinstance(special_token_value, list):
                                for special_tok in special_token_value:
                                    if special_tok: output = output.replace(special_tok, "")
                            else:
                                if special_token_value: output = output.replace(special_token_value, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                
                except Exception as e: # Catch generic vLLM errors
                    print(f"ERROR question ID: {question['question_id']}, error: {e}")
                    output = "ERROR"

                conv.update_last_message(output)
                turns.append(output)

            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            # Handle potential empty lines or malformed JSON
            try:
                data = json.loads(l)
                qid = data["question_id"]
                answers[qid] = l
            except json.JSONDecodeError:
                print(f"Skipping malformed line in {answer_file}: {l.strip()}")
                continue


    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model", # This will map to tensor_parallel_size for vLLM
        type=int,
        default=1,
        help="The number of GPUs per model instance (tensor_parallel_size for vLLM).",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs to use (for Ray workers)."
    )
    parser.add_argument(
        "--gpu-memory-utilization", # New argument for vLLM
        type=float,
        default=0.9,
        help="The fraction of GPU memory to be used by the vLLM engine (0 to 1).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"], # vLLM dtypes
        help="Override the default dtype. If not set, vLLM uses 'auto'.",
        default="auto", # Changed default for vLLM
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    # --load-8bit and --cpu-offloading are not applicable to vLLM in the same way.
    # --max-gpu-memory is replaced by --gpu-memory-utilization for vLLM.

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray
        ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        # Ensure the directory for answers exists
        answer_dir = f"data/{args.bench_name}/model_answer/"
        os.makedirs(answer_dir, exist_ok=True)
        answer_file = os.path.join(answer_dir, f"{args.model_id}.jsonl")


    print(f"Output to {answer_file}")

    # Ensure dtype is a string, as processed by argparse. No str_to_torch_dtype needed.
    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype, # Pass dtype string directly
        revision=args.revision,
    )

    reorg_answer_file(answer_file)
