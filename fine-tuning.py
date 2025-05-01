from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model , PeftModel
import torch

# 1. Configuração inicial
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# 2. Carregar e preparar dados
dataset = load_dataset("json", data_files="base_final_para_finetuning.jsonl")

def format_function(examples):
    # Formatar para texto puro (alternativa ao chat template)
    texts = []
    for msg_list in examples["messages"]:
        text = ""
        for msg in msg_list:
            text += f"{msg['role']}: {msg['content']}\n"
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_function, batched=True)

# 3. Carregar modelo e tokenizer
model_path = "models/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# Configuração de quantização
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    use_safetensors=True,
    trust_remote_code=True,
    torch_dtype=torch.float16, 
)

print("Camadas do modelo:")
for name, module in model.named_modules():
    print(name)

# 4. Tokenização correta (cria automaticamente input_ids e attention_mask)
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    # Cria labels para language modeling (mesmo que input_ids)
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["messages", "text"]  # Remove colunas originais
)

# 5. Configuração LoRA
# Configuração LoRA otimizada
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],  # Módulos exatos
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["embed_tokens", "norm", "lm_head"]  # Para melhor adaptação
)

model = get_peft_model(model, peft_config)


training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,  # Reduzido para caber na memória da GPU
    gradient_accumulation_steps=8,  # Acumula para simular batch maior
    learning_rate=2e-5,
    logging_steps=10,
    max_steps=300,
    warmup_steps=30,
    save_strategy="steps",
    save_steps=200,
    fp16=False,  # Habilita treino em float16 (ótimo para GPUs da série 40)
    bf16=True,  # Sua 4060Ti não suporta bfloat16
    optim="adamw_torch",  # COMPATÍVEL com fp16
    remove_unused_columns=True,
    gradient_checkpointing=True,  # Ajuda a economizar memória
    logging_dir="./logs",  # Local para logs do TensorBoard (opcional)
    report_to="none",  # Evita erro se você não usar wandb ou TensorBoard
)


trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parâmetros treináveis: {trainable_params}")

if trainable_params == 0:
    raise ValueError("Nenhum parâmetro treinável encontrado! Verifique target_modules do LoRA")
# 7. Data collator para language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Para modelos causais (não masked LM)
)

# 8. Criar Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# 9. Treinar
print("Iniciando treinamento...")
trainer.train()

# 10. Salvar modelo
model.save_pretrained("modelo_ajustado")
print("Modelo ajustado salvo com sucesso!")


# 1. Carregue o modelo base CORRETAMENTE (force um único arquivo se necessário)
model = AutoModelForCausalLM.from_pretrained(
    "models/",
    device_map="auto",
    use_safetensors=True,
    torch_dtype="auto",  # ou torch.float16 para GPU
)

# 2. Carregue o adaptador LoRA (deve funcionar, pois você tem adapter_model.safetensors)
model = PeftModel.from_pretrained(model, "modelo_ajustado")

# 3. Funda o LoRA no modelo base
model = model.merge_and_unload()

# 4. Salve o modelo final
model.save_pretrained("modelo_fundido", safe_serialization=True)