from llama_cpp import Llama
from datasets import load_dataset
import json
import logging
import re
from tqdm import tqdm

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="casos_gerados.log"
)
logger = logging.getLogger(__name__)

def preprocess_content(content):
    """Padroniza o conteúdo para extração do artigo."""
    content = content.replace("Artigo ", "artigo ")
    content = content.replace("Art. ", "artigo ")
    content = content.replace("art. ", "artigo ")
    return content.strip()

def extract_article_info(message_pair):
    """Extrai informações do artigo do formato de mensagem de forma robusta."""
    try:
        user_content = preprocess_content(message_pair[0]["content"])
        assistant_content = message_pair[1]["content"]
        
        # Padrões para extrair o número do artigo
        patterns = [
            r"artigo\s*(\d+)",         # "artigo 3"
            r"artigo\s*:\s*(\d+)",     # "artigo: 3"
            r"artigo\s*-?\s*(\d+)",    # "artigo-3" ou "artigo - 3"
            r"artigo\s*(\d+)\D",       # "artigo 3 do"
            r"\b(\d+)\s*-\s*",         # "3 - "
            r"artigo\s*n[º°]\s*(\d+)", # "artigo nº 3"
            r"artigo\s*(\d+)",         # fallback genérico
        ]
        
        article_number = None
        for pattern in patterns:
            match = re.search(pattern, user_content, re.IGNORECASE)
            if match:
                article_number = match.group(1)
                break
        
        if article_number and article_number.isdigit():
            return f"artigo {article_number}", assistant_content
        else:
            # Fallback: tenta extrair o primeiro número encontrado
            numbers = re.findall(r'\d+', user_content)
            if numbers:
                return f"artigo {numbers[0]}", assistant_content
            
            logger.warning(f"Não foi possível extrair número do artigo de: {user_content}")
            return None, None  # Retorna None para filtrar casos inválidos
            
    except Exception as e:
        logger.error(f"Erro ao extrair informações do artigo: {str(e)}")
        return None, None

def validate_article_number(article_num):
    """Verifica se o número extraído é válido."""
    if not article_num:
        return False
    return article_num.replace("artigo ", "").isdigit()

def generate_text(llm, artigo, artigo_descricao):
    """Gera texto usando o modelo Llama com prompt otimizado."""
    try:
        prompt = f"""
        Você é um especialista em direito penal brasileiro. 
        Gere UM ÚNICO caso hipotético realista para ilustrar a aplicação do {artigo} do Código Penal.
        
        **Formato exigido:**
        1. Contexto (local, envolvidos)
        2. Ação que configura o crime
        3. Detalhes relevantes
        4. Pergunta clara sobre qual artigo se aplica
        
        **Exemplo:**
        "Em uma cidade do interior, um comerciante adulterou produtos..."
        "Qual artigo do Código Penal se aplica a essa situação?"
        
        **Artigo {artigo}:**
        {artigo_descricao}
        """
        
        output = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        return output["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Erro ao gerar texto para {artigo}: {str(e)}")
        return None

def validate_generated_text(text):
    """Valida o texto gerado pelo LLM."""
    if not text or len(text.strip()) < 50:
        raise ValueError("Texto gerado inválido ou muito curto")
    return text.strip()

def process_dataset(llm, dataset):
    """Processa o dataset e gera casos hipotéticos."""
    casos_gerados = []
    skipped = 0
    
    for item in tqdm(dataset["train"], desc="Processando artigos"):
        try:
            messages = item["messages"]
            if len(messages) < 2:
                skipped += 1
                continue
                
            artigo, descricao = extract_article_info(messages)
            if not artigo or not validate_article_number(artigo):
                skipped += 1
                continue
                
            caso_gerado = generate_text(llm, artigo, descricao)
            if not caso_gerado:
                skipped += 1
                continue
                
            caso_validado = validate_generated_text(caso_gerado)
            
            casos_gerados.append({
                "messages": [
                    {"role": "user", "content": caso_validado},
                    {"role": "assistant", "content": f"Esse caso se enquadra no {artigo}, que define: {descricao}"}
                ]
            })
            
        except Exception as e:
            logger.error(f"Erro ao processar item: {str(e)}")
            skipped += 1
            continue
    
    logger.info(f"Artigos processados: {len(casos_gerados)} | Skipped: {skipped}")
    return casos_gerados

def save_to_jsonl(data, output_file):
    """Salva os dados em formato JSONL."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for item in data:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + "\n")
        logger.info(f"Dados salvos em {output_file}")
    except Exception as e:
        logger.error(f"Erro ao salvar arquivo: {str(e)}")
        raise

def main():
    try:
        logger.info("Inicializando modelo LLM...")
        llm = Llama.from_pretrained(
         repo_id="Inza124/Llama3.2_3b",
         filename="Llama3.2-maIN.gguf",
         n_gpu_layers=-1, # Uncomment to use GPU acceleration
         seed=1337, # Uncomment to set a specific seed
         n_ctx=2048, # Uncomment to increase the context window
        )
        
        logger.info("Carregando dataset...")
        dataset = load_dataset("json", data_files="data/legislacao_pronta/codigo_penal.jsonl")
        
        logger.info("Gerando casos hipotéticos...")
        casos_gerados = process_dataset(llm, dataset)
        
        save_to_jsonl(casos_gerados, "casos_penais_gerados.jsonl")
        
        logger.info(f"Processo concluído. Casos gerados: {len(casos_gerados)}")
        
    except Exception as e:
        logger.error(f"Falha crítica: {str(e)}")
        raise

if __name__ == "__main__":
    main()