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
    "bf16": True,  # Pode manter pois a 4060 Ti suporta bfloat16
    "do_eval": True,
    "learning_rate": 2e-5,
    "logging_steps": 20,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 3,
    "output_dir": "./finetuned_model",
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 1,  # Seguro para 8-16GB VRAM
    "gradient_accumulation_steps": 4,  # Simula batch size maior
    "remove_unused_columns": True,
    "save_steps": 500,
    "eval_steps": 500,
    "save_strategy": "steps",
    "eval_strategy": "steps",
    "save_total_limit": 2,
    "seed": 42,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "warmup_ratio": 0.1,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "load_best_model_at_end": True,
    "learning_rate": 1.5e-5,  # Reduzir ligeiramente
    "warmup_ratio": 0.15,     # Mais warmup
    "lr_scheduler_type": "cosine_with_restarts",
    "num_train_epochs": 5,
}


peft_config = {
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": "all-linear",
    "modules_to_save": None,
    "target_modules": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
}

train_args = TrainingArguments(**training_config)
peft_conf = LoraConfig(**peft_config)

# ======================
# Logging
# ======================
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = train_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

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
model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="cuda",
    torch_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'

# ======================
# Dataset
# ======================
dataset_path = "base_final_para_finetuning.jsonl"


def apply_chat_template(example, tokenizer=None):
    # Removido o json.loads
    messages = example["messages"]
    chat = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            chat += f"Usuário: {content}\n"
        elif role == "assistant":
            chat += f"Assistente: {content}\n"
    return chat.strip()



def load_and_prepare_dataset(path, tokenizer):
    # Carrega o dataset
    raw_dataset = load_dataset("json", data_files=path, split="train")
    
    # Processa para ter certeza que está no formato correto
    def ensure_format(example):
        if "messages" not in example:
            if "prompt" in example and "completion" in example:
                example["messages"] = [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["completion"]}
                ]
            else:
                raise ValueError("Formato inválido - exemplo deve conter 'messages' ou 'prompt'/'completion'")
        return example
    
    processed = raw_dataset.map(
        ensure_format,
        num_proc=min(multiprocessing.cpu_count(), 4),
        desc="Garantindo formato correto"
    )
    
    return processed.train_test_split(test_size=0.1, seed=42)

# ======================
# Treinamento
# ======================
def train_model():
    set_seed(training_config["seed"])
    datasets = load_and_prepare_dataset(dataset_path, tokenizer)
    train_dataset = datasets["train"]
    eval_dataset = datasets["test"]

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_conf,
        processing_class=tokenizer,  # Aqui passamos o tokenizer
        formatting_func=lambda batch: [apply_chat_template({"messages": msgs}) for msgs in batch["messages"]]

    )

    train_result = trainer.train()
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
