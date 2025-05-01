import json

# Função para melhorar a resposta
def melhorar_resposta(artigo, resposta_original):
    resposta_melhorada = (
        f"De acordo com o Código Penal Brasileiro atualizado em 2024, o artigo {artigo} dispõe que:\n\n"
        f"{resposta_original.strip()}."
    )
    return resposta_melhorada

# Caminho dos arquivos
input_file = "data/legislacao_pronta/codigo_penal.jsonl"      # seu arquivo atual
output_file = "base_atualizada.jsonl" # novo arquivo gerado

# Lista para guardar os novos dados
novas_mensagens = []

# Abrir e processar o arquivo original
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        user_content = item["messages"][0]["content"]
        assistant_content = item["messages"][1]["content"]

        # Tentar extrair o número do artigo da pergunta
        artigo_num = ""
        if "art" in user_content.lower():
            partes = user_content.split()
            for p in partes:
                if p.isdigit():
                    artigo_num = p
                    break

        # Melhorar a resposta
        resposta_melhorada = melhorar_resposta(artigo_num, assistant_content)

        # Montar a nova mensagem
        novas_mensagens.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": resposta_melhorada}
            ]
        })

# Salvar o novo arquivo JSONL
with open(output_file, "w", encoding="utf-8") as f_out:
    for item in novas_mensagens:
        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Base atualizada salva como '{output_file}' com {len(novas_mensagens)} exemplos!")
