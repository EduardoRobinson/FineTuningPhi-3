import json
import random
import re

# Funções auxiliares
def extrair_numero_artigo(texto):
    """Extrai o número do artigo a partir da pergunta do usuário"""
    match = re.search(r"art(?:igo)?\s*(\d+)", texto, re.IGNORECASE)
    if match:
        return match.group(1)
    return ""

def gerar_caso_de_uso(artigo_num):
    """Gera uma pergunta prática a partir do número do artigo"""
    exemplos = [
        f"Em uma situação prática, como é aplicado o artigo {artigo_num} do Código Penal Brasileiro?",
        f"Pode me dar um exemplo de quando o artigo {artigo_num} é usado na prática?",
        f"Quando ocorre uma situação relacionada ao artigo {artigo_num}, como a lei trata?",
        f"Se alguém violar o artigo {artigo_num} do Código Penal, o que acontece?",
        f"Explique um caso real onde o artigo {artigo_num} do Código Penal é aplicado."
    ]
    return random.choice(exemplos)

def gerar_resposta_pratica(artigo_num, conteudo_assistant):
    """Gera uma resposta de caso prático a partir do conteúdo do artigo"""
    resposta = (
        f"Claro! Suponha que ocorra uma situação relacionada ao artigo {artigo_num}.\n\n"
        f"Conforme o Código Penal Brasileiro atualizado em 2024, esse artigo determina:\n\n"
        f"{conteudo_assistant.strip()}\n\n"
        f"A lei será aplicada conforme descrito, considerando os detalhes do caso."
    )
    return resposta

# Caminhos dos arquivos
input_file = "base_atualizada.jsonl"  # Arquivo que geramos antes
output_file = "casos_de_uso_gerados.jsonl"

# Lista para armazenar os novos exemplos
casos_de_uso = []

# Ler o arquivo existente
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        user_content = item["messages"][0]["content"]
        assistant_content = item["messages"][1]["content"]

        # Extrair o número do artigo
        artigo_num = extrair_numero_artigo(user_content)
        if artigo_num == "":
            continue  # pula se não encontrar artigo

        # Gerar o novo par de mensagem
        pergunta_caso = gerar_caso_de_uso(artigo_num)
        resposta_caso = gerar_resposta_pratica(artigo_num, assistant_content)

        casos_de_uso.append({
            "messages": [
                {"role": "user", "content": pergunta_caso},
                {"role": "assistant", "content": resposta_caso}
            ]
        })

# Salvar novo arquivo JSONL
with open(output_file, "w", encoding="utf-8") as f_out:
    for item in casos_de_uso:
        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Gerados {len(casos_de_uso)} casos de uso baseados na 'base_atualizada.jsonl' e salvos em '{output_file}'!")
