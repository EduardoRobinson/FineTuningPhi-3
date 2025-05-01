import sys
import logging
import multiprocessing

from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
    EarlyStoppingCallback
)
from trl import SFTTrainer
import datasets
import transformers
import torch
import json

# ======================
# Configurações
# ======================
training_config = {
    "bf16": True,
    "do_eval": True,
    "learning_rate": 3e-5,
    "logging_steps": 20,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine_with_restarts",
    "num_train_epochs": 10,
    "output_dir": "./finetuned_model",
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 2,
    "remove_unused_columns": True,
    "save_steps": 500,
    "eval_steps": 500,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "save_total_limit": 2,
    "seed": 42,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "warmup_ratio": 0.15,
    "metric_for_best_model": "eval_mean_token_accuracy",
    "greater_is_better": False,
    "load_best_model_at_end": True,
    "optim": "adamw_torch",  # Usar AdamW com correção de peso
    "weight_decay": 0.01,  # Adicionar decay para regularização
    "max_grad_norm": 1.0, # Limitar o gradiente para evitar explosão
}

peft_config = {
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.1,
    "bias": "lora_only",
    "task_type": "CAUSAL_LM",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}

# ======================
# Logging
# ======================
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ======================
# Modelo e Tokenizer
# ======================
checkpoint_path = "models/"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

def get_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    tokenizer.padding_side = 'right'
    
    return model, tokenizer

# ======================
# Dataset
# ======================
dataset_path = "base_final_para_finetuning.jsonl"

def format_chat_template(example):
    messages = example["messages"]
    formatted_text = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            formatted_text += f"Usuário: {content}\n"
        elif role == "assistant":
            formatted_text += f"Assistente: {content}\n"
    return {"text": formatted_text} 

def load_and_prepare_dataset(path):
    dataset = load_dataset("json", data_files=path, split="train")
    
    # Format the dataset
    dataset = dataset.map(
        format_chat_template,
        remove_columns=["messages"],  # Remove original column after formatting
        num_proc=min(multiprocessing.cpu_count(), 4)
    )
    
    return dataset.train_test_split(test_size=0.1, seed=42)

# ======================
# Treinamento
# ======================
def train_model():
    set_seed(training_config["seed"])
    
    # Initialize model and tokenizer
    model, tokenizer = get_model_and_tokenizer()
    
    # Load and prepare dataset
    dataset = load_and_prepare_dataset(dataset_path)
    
    # Initialize training arguments
    train_args = TrainingArguments(**training_config)
    
    # Initialize PEFT config
    peft_conf = LoraConfig(**peft_config)
    
    # Set up logging
    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_conf,
    )
    
    # Train the model
    train_result = trainer.train()
    
    # Save everything
    trainer.save_model(train_args.output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

# ======================
# Execução principal
# ======================
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    train_model()