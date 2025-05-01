import streamlit as st
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# Caminho do modelo fine-tunado
MODEL_DIR = Path("C:/Users/miojo/Documents/TCC-BCC/models")

# Configuração da página
st.set_page_config(page_title="Phi-3 Chat", page_icon="🤖")

@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            device_map="auto",  # usa a melhor opção disponível
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            local_files_only=True
        )
        return model.eval(), tokenizer
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {str(e)}")
        return None, None

def format_prompt(prompt):
    # Pode ser ajustado conforme seu fine-tuning (e.g., se treinou com templates tipo chatML)
    return f"<|user|>\n{prompt}\n<|assistant|>\n"

def stream_response(prompt, model, tokenizer, max_new_tokens, temperature):
    formatted_prompt = format_prompt(prompt)

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_args = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": True,
        "top_p": 0.9,
        "eos_token_id": tokenizer.eos_token_id or tokenizer.pad_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    thread = Thread(target=model.generate, kwargs=generation_args)
    thread.start()

    partial_text = ""
    with st.empty():
        for token in streamer:
            partial_text += token
            st.markdown(partial_text + "▌")  # efeito de digitação

    return partial_text.strip()

def main():
    model, tokenizer = load_model()

    st.title("🤖 Chat com Phi-3 Ajustado")

    # Sidebar
    with st.sidebar:
        st.header("Configurações")
        max_new_tokens = st.slider("Máximo de tokens gerados", 64, 2048, 512)
        temperature = st.slider("Temperatura", 0.1, 2.0, 0.7)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibe mensagens anteriores
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input do usuário
    if user_input := st.chat_input("Digite sua mensagem..."):
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            if model and tokenizer:
                response = stream_response(
                    prompt=user_input,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
            else:
                response = "Erro ao carregar o modelo."

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
