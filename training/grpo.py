import torch
from tqdm import tqdm
import logging
import os
import glob
import argparse
import wandb

from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from data_loader import create_dataset
from utils import set_seed, load_model, check_existence
from config import parse_args
from chat_templates import QWEN_CHAT_TEMPLATE, LLAMA_CHAT_TEMPLATE, OLMO_CHAT_TEMPLATE

def main():
    config = parse_args()

    if config.reasoning:
        assert "format" in config.reward_funcs, "format reward function is required when reasoning is True"

    set_seed(config.seed)
    logging.basicConfig(filemode='w', level=logging.INFO)
    logging.info("Config:")
    logging.info('\n\t'.join(f'{k}={v}' for k, v in vars(config).items()))

    access_token = os.getenv("HF_TOKEN")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, cache_dir=config.cache_dir, token=access_token)
    if tokenizer.chat_template is None:
        if "qwen" in config.model_path.lower():
            print("No chat template found for Qwen, using default template")
            tokenizer.chat_template = QWEN_CHAT_TEMPLATE
        elif "llama" in config.model_path.lower():
            print("No chat template found for Llama, using default template")
            tokenizer.chat_template = LLAMA_CHAT_TEMPLATE
        elif "olmo" in config.model_path.lower():
            print("No chat template found for OLMo, using default template")
            tokenizer.chat_template = OLMO_CHAT_TEMPLATE
        else:
            raise ValueError(f"Unsupported model: {config.model_path}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = config.model_max_length
    tokenizer.padding_side = 'left'
    max_prompt_length = config.max_prompt_length
    if max_prompt_length == -1:
        max_prompt_length = tokenizer.model_max_length

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(model_path=config.model_path, cache_dir=config.cache_dir, access_token=access_token)
    logging.info("Loaded model {} on device {}".format(config.model_path, device))

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False

    ds = create_dataset(
        data_path=config.data_path,
        split=config.train_split,
        cache_dir=config.cache_dir,
        streaming=config.streaming,
        shuffle=config.shuffle,
        load_from_disk=config.load_from_disk,
        tokenizer=tokenizer
    )

    reward_funcs = ds.get_reward_funcs(config.reward_funcs)
    print(f"Using reward functions: {reward_funcs}")

    ds.add_conversation_format(reasoning=config.reasoning)
    ds = ds.get_dataset()
    ds = ds.remove_columns(["messages"])  # https://github.com/huggingface/trl/blob/main/trl/data_utils.py#L81
    
    ckpt_dir = config.ckpt_dir
    last_checkpoint = None
    
    if check_existence(ckpt_dir, isDir=True):
        checkpoints = glob.glob(os.path.join(ckpt_dir, "checkpoint-*"))
        last_checkpoint = max(checkpoints, key=os.path.getmtime) if checkpoints else None
        logging.info(f"Found checkpoint from previous run(s).\n\tResuming from checkpoint: {last_checkpoint}")
    
    wandb.init(
        project=config.wandb_project,
        name=config.run_name,
        config=vars(config),
        save_code=True
    )

    training_args = GRPOConfig(
        run_name=config.run_name,
        output_dir=ckpt_dir,
        log_level="info",
        logging_steps=1,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        num_generations=config.num_generations,  # GRPO group size
        max_prompt_length=config.max_prompt_length,  # max length of the prompt
        max_completion_length=config.max_completion_length,  # max length of the completion
        temperature=config.temperature,
        report_to="wandb",
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=config.bf16,
        num_iterations=config.num_iterations,
        torch_empty_cache_steps=20,
        log_completions=True,
        deepspeed=config.deepspeed,
        local_rank=config.local_rank,
        shuffle_dataset=config.shuffle
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=ds
    )
    trainer.tokenizer.padding_side = "left"
    trainer.processing_class.padding_side = "left"

    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    wandb.finish()

if __name__ == "__main__":
    main()
