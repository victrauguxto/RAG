"""Calcula as métricas do estudo a partir do gabarito e dos resultados do comparar.py.

    python avaliar.py resultados/estudo_3_rodadas.jsonl

Busca (refaz a busca de cada pergunta; só usa o modelo de embedding):
  - hit@5 por página: alguma página do gabarito entre os 5 trechos recuperados
  - MRR por página: 1 / posição do primeiro trecho de uma página do gabarito
  - hit@5 por trecho: algum trecho recuperado contém todos os fatos da resposta
Geração (lê as respostas gravadas):
  - acerto por fatos-chave: a resposta contém todos os fatos do gabarito
  - consistência: a pergunta teve o mesmo veredito em todas as rodadas
  - latência e tokens médios

O acerto por fatos não detecta rótulos trocados nem citação de página errada;
esses casos foram revisados à mão (veja o README).
"""
import json, re, sys
from collections import defaultdict
from statistics import mean

import main
import rag_langchain

GABARITO = "estudo/perguntas.txt"
VERSOES = {"do_zero": main, "langchain": rag_langchain}


def normalizar(texto):
    return re.sub(r"\s+", " ", texto.replace("–", "-")).lower()


def contem(texto, fatos):
    texto = normalizar(texto)
    return all(f in texto for f in fatos)


def ler_gabarito():
    gabarito = {}
    with open(GABARITO, encoding="utf-8") as f:
        for linha in f:
            if not linha.strip() or linha.startswith("#"):
                continue
            pergunta, paginas, fatos = [p.strip() for p in linha.split("|")]
            gabarito[pergunta] = {
                "paginas": [int(p) for p in paginas.replace(",", " ").split()],
                "fatos": [normalizar(x.strip()) for x in fatos.split(";")],
            }
    return gabarito


def avaliar_busca(gabarito):
    metricas = {}
    for nome, modulo in VERSOES.items():
        hits, rr, hits_trecho = [], [], []
        for pergunta, g in gabarito.items():
            resultados = modulo.buscar(pergunta)
            paginas = [c["pagina"] for c, _ in resultados]
            posicao = next((i for i, p in enumerate(paginas, 1) if p in g["paginas"]), None)
            hits.append(posicao is not None)
            rr.append(1 / posicao if posicao else 0)
            hits_trecho.append(any(contem(c["texto"], g["fatos"]) for c, _ in resultados))
        metricas[nome] = {
            "hit@5 (página)": f"{sum(hits)}/{len(hits)}",
            "MRR (página)": f"{mean(rr):.2f}",
            "hit@5 (trecho)": f"{sum(hits_trecho)}/{len(hits_trecho)}",
        }
    return metricas


def avaliar_geracao(arquivo, gabarito):
    with open(arquivo, encoding="utf-8") as f:
        linhas = [json.loads(l) for l in f]
    por_pergunta = vereditos_por_pergunta(linhas, gabarito)
    metricas = {}
    for nome in VERSOES:
        vereditos = [v[nome] for v in por_pergunta.values()]
        acertos = sum(sum(v) for v in vereditos)
        total = sum(len(v) for v in vereditos)
        consistentes = sum(len(set(v)) == 1 for v in vereditos)
        registros = [l[nome] for l in linhas]
        metricas[nome] = {
            "acerto por fatos-chave": f"{acertos}/{total} ({acertos / total:.0%})",
            "consistência entre rodadas": f"{consistentes}/{len(vereditos)} perguntas",
            "latência média (s)": f"{mean(r['tempo_total_s'] for r in registros):.1f}",
        }
        if "tokens_entrada" in registros[0]:
            metricas[nome]["tokens de entrada / pergunta"] = f"{mean(r['tokens_entrada'] for r in registros):.0f}"
            metricas[nome]["tokens de saída / pergunta"] = f"{mean(r['tokens_saida'] for r in registros):.0f}"
            metricas[nome]["tokens/s"] = f"{mean(r['tokens_por_s'] for r in registros):.1f}"
    return metricas, por_pergunta


def vereditos_por_pergunta(linhas, gabarito):
    tabela = defaultdict(lambda: defaultdict(list))
    for l in linhas:
        for nome in VERSOES:
            tabela[l["pergunta"]][nome].append(contem(l[nome]["resposta"], gabarito[l["pergunta"]]["fatos"]))
    return tabela


def imprimir_tabela(titulo, metricas):
    print(f"\n### {titulo}\n")
    print("| Métrica | Do zero | LangChain |\n|---|---|---|")
    for chave in metricas["do_zero"]:
        print(f"| {chave} | {metricas['do_zero'][chave]} | {metricas['langchain'][chave]} |")


if __name__ == "__main__":
    arquivo = sys.argv[1] if len(sys.argv) > 1 else "resultados/estudo_3_rodadas.jsonl"
    gabarito = ler_gabarito()
    imprimir_tabela("Busca", avaliar_busca(gabarito))
    geracao, por_pergunta = avaliar_geracao(arquivo, gabarito)
    imprimir_tabela("Geração e operação", geracao)

    print("\n### Acerto por pergunta (fatos-chave)\n")
    print("| Pergunta | Do zero | LangChain |\n|---|---|---|")
    for pergunta, v in por_pergunta.items():
        print(f"| {pergunta} | {sum(v['do_zero'])}/{len(v['do_zero'])} "
              f"| {sum(v['langchain'])}/{len(v['langchain'])} |")
