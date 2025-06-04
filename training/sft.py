import torch
from tqdm import tqdm
import logging
import os
import glob
import wandb

from transformers import AutoTokenizer
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
from chat_templates import LLAMA_CHAT_TEMPLATE

from data_loader import create_dataset
from utils import set_seed, load_model, check_existence
from config import parse_args

def format_data(example):
    """
    Converts the data format from:
    {
        'id': '...',
        'source': '...',
        'messages': [{'content': '...', 'role': 'user'}, {'content': '...', 'role': 'assistant'}]
    }
    to:
    {
        'id': '...',
        'source': '...',
        'messages': [...],
        'prompt': [{'role': 'user', 'content': '...'}],
        'completion': [{'role': 'assistant', 'content': '...'}]
    }
    """
    messages = example['messages']
    
    # Find user and assistant messages
    user_messages = [msg for msg in messages if msg['role'] == 'user']
    assistant_messages = [msg for msg in messages if msg['role'] == 'assistant']
    
    if user_messages and assistant_messages:
        # Create new dictionaries with the exact order of keys as requested
        user_msg = user_messages[0]
        assistant_msg = assistant_messages[0]
        
        # Create new dictionaries with role before content
        formatted_user_msg = {'role': user_msg['role'], 'content': user_msg['content']}
        formatted_assistant_msg = {'role': assistant_msg['role'], 'content': assistant_msg['content']}
        
        example['prompt'] = [formatted_user_msg]
        example['completion'] = [formatted_assistant_msg]
    else:
        # Handle edge case where messages don't follow the expected format
        example['prompt'] = []
        example['completion'] = []
    
    return example

def main():
    config = parse_args()

    set_seed(config.seed)
    logging.basicConfig(filemode='w', level=logging.INFO)
    logging.info("Config:")
    logging.info('\n\t'.join(f'{k}={v}' for k, v in vars(config).items()))

    access_token = os.getenv("HF_TOKEN")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(model_path=config.model_path, cache_dir=config.cache_dir, access_token=access_token)
    logging.info("Loaded model {} on device {}".format(config.model_path, device))
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, cache_dir=config.cache_dir, token=access_token)

    tokenizer.add_special_tokens({"pad_token":"[PAD]"})
    tokenizer.pad_token = "[PAD]"
    tokenizer.model_max_length = config.model_max_length
    model.resize_token_embeddings(len(tokenizer))
    
    if "llama" in config.model_path.lower():
        tokenizer.chat_template = LLAMA_CHAT_TEMPLATE
        tokenizer.eos_token = '<|eot_id|>'
        model.generation_config.eos_token_id = [128001, 128009]
    elif "qwen" in config.model_path.lower():
        tokenizer.eos_token = '<|im_end|>'
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = [151645, 151643]
    else:
        print("Unknown model type")
    
    max_prompt_length = config.max_prompt_length
    if max_prompt_length == -1:
        max_prompt_length = tokenizer.model_max_length
    
    model.config.use_cache = False

    dataset = load_from_disk(config.data_path)

    ds = dataset[config.train_split]
    # Apply the format_data function to the dataset
    ds = ds.map(format_data)
    # Remove the messages column after formatting
    ds = ds.remove_columns(['messages'])
    logging.info(f"Formatted {len(ds)} training examples")
    
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

    training_args = SFTConfig(
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
        max_steps=config.max_steps,
        report_to="wandb",
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=config.bf16,
        torch_compile=False,
        deepspeed=config.deepspeed,
        local_rank=config.local_rank,
        packing=config.packing,  # Don't pack multiple conversations into one sequence
        dataset_text_field=config.dataset_text_field,  # Use messages field which contains the conversation,
        torch_empty_cache_steps=20,
        gradient_checkpointing=True,
        max_length=1024,
        eos_token=tokenizer.eos_token,
        completion_only_loss=True
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=ds
    )

    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    wandb.finish()

if __name__ == "__main__":
    main()