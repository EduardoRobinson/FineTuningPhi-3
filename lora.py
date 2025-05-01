from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. Carregue o modelo base CORRETAMENTE (force um único arquivo se necessário)
model = AutoModelForCausalLM.from_pretrained(
    "models/",
    device_map="auto",
    use_safetensors=True,
    torch_dtype="auto",  # ou torch.float16 para GPU
)

# 2. Carregue o adaptador LoRA (deve funcionar, pois você tem adapter_model.safetensors)
model = PeftModel.from_pretrained(model, "finetuned_model")

# 3. Funda o LoRA no modelo base
model = model.merge_and_unload()

# 4. Salve o modelo final
model.save_pretrained("modelo_fundido", safe_serialization=True)