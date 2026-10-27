import json, sys
import numpy as np
import requests
from pypdf import PdfReader

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
LLM = "qwen3.5:4b"

# compartilhado com rag_langchain.py para a comparação ser justa
PROMPT = """Você é um assistente que responde dúvidas sobre o manual de uma moto.
Use APENAS os trechos abaixo. Cite as páginas usadas.
Se a resposta não estiver nos trechos, diga que não encontrou no manual.

TRECHOS:
{contexto}

PERGUNTA: {pergunta}"""


def extrair_paginas(caminho_pdf):
    reader = PdfReader(caminho_pdf)
    return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]


def chunkar(paginas, tamanho=800, sobreposicao=150):
    chunks = []
    for pagina, texto in paginas:
        texto = " ".join(texto.split())
        inicio = 0
        while inicio < len(texto):
            chunks.append({"pagina": pagina, "texto": texto[inicio:inicio + tamanho]})
            inicio += tamanho - sobreposicao
    return chunks


def embed(textos, lote=32):
    vetores = []
    for i in range(0, len(textos), lote):
        r = requests.post(f"{OLLAMA}/api/embed",
                          json={"model": EMBED_MODEL, "input": textos[i:i + lote]})
        r.raise_for_status()
        vetores.extend(r.json()["embeddings"])
    v = np.array(vetores, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)  # normaliza → cosseno = produto escalar


def indexar(caminho_pdf):
    chunks = chunkar(extrair_paginas(caminho_pdf))
    vetores = embed([c["texto"] for c in chunks])
    np.save("vetores.npy", vetores)
    json.dump(chunks, open("chunks.json", "w"), ensure_ascii=False)
    print(f"{len(chunks)} chunks indexados.")


def buscar(pergunta, k=5):
    vetores = np.load("vetores.npy")
    chunks = json.load(open("chunks.json"))
    q = embed([pergunta])[0]
    scores = vetores @ q
    melhores = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in melhores]


def responder(pergunta):
    texto, resultados = gerar_resposta(pergunta)
    print(texto)
    print("\nFontes:", sorted({c["pagina"] for c, _ in resultados}))


def gerar_resposta(pergunta):
    resultados = buscar(pergunta)
    contexto = "\n\n".join(f"[página {c['pagina']}]\n{c['texto']}" for c, _ in resultados)
    prompt = PROMPT.format(contexto=contexto, pergunta=pergunta)
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": LLM,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.2},
    })
    r.raise_for_status()
    return r.json()["message"]["content"], resultados


if __name__ == "__main__":
    comando = input("Gostaria de indexar ou perguntar algo? ( 1 para indexar, 2 para perguntar): ")
    if comando == "1":
        indexar(input("Digite o caminho do PDF: "))
    elif comando == "2":
        responder(input("Qual é a sua pergunta? "))
    else:
        print("Comando inválido. Escolha '1' ou '2'.")