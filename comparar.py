"""Faz a mesma pergunta ao RAG do zero (main.py) e ao RAG com LangChain e compara.

Cada pergunta é registrada em um arquivo JSONL (padrão: resultados/comparacao.jsonl).
Uso:
    python comparar.py                                   # modo interativo
    python comparar.py estudo/perguntas.txt              # uma pergunta por linha
    python comparar.py estudo/perguntas.txt --rodadas 3 --saida resultados/estudo.jsonl

Opcionalmente, informe as páginas corretas depois de um "|":
    Qual a folga correta da corrente? | 71
Assim o resultado registra se a busca trouxe alguma delas no top-k.
"""
import argparse, json, os, time
from datetime import datetime

import main
import rag_langchain

SAIDA = "resultados/comparacao.jsonl"
VERSOES = {"do_zero": main, "langchain": rag_langchain}


def rodar(modulo, pergunta):
    # busca medida à parte só para o tempo; a resposta refaz a busca internamente
    inicio = time.perf_counter()
    modulo.buscar(pergunta)
    t_busca = time.perf_counter() - inicio

    inicio = time.perf_counter()
    texto, resultados, uso = modulo.gerar_resposta(pergunta)
    t_total = time.perf_counter() - inicio
    return {
        "resposta": texto,
        "paginas": [c["pagina"] for c, _ in resultados],
        "scores": [round(s, 3) for _, s in resultados],
        "tempo_busca_s": round(t_busca, 2),
        "tempo_total_s": round(t_total, 2),
        **uso,
    }


def ler_linha(linha):
    """'pergunta | 64, 65 | fatos' -> ('pergunta', [64, 65]); os fatos são usados pelo avaliar.py"""
    partes = [p.strip() for p in linha.split("|")]
    esperadas = [int(p) for p in partes[1].replace(",", " ").split()] if len(partes) > 1 else []
    return partes[0], esperadas


def comparar(linha, rodada=1):
    pergunta, esperadas = ler_linha(linha)
    registro = {"data": datetime.now().isoformat(timespec="seconds"), "rodada": rodada,
                "pergunta": pergunta, "paginas_esperadas": esperadas}
    for nome, modulo in VERSOES.items():
        r = rodar(modulo, pergunta)
        if esperadas:
            r["acertou_busca"] = bool(set(esperadas) & set(r["paginas"]))
        registro[nome] = r
        print(f"\n===== {nome.upper()}  ({r['tempo_total_s']}s) =====")
        print(r["resposta"])
        print(f"Páginas recuperadas: {r['paginas']}  scores: {r['scores']}")
        print(f"Tokens: {r['tokens_entrada']} entrada, {r['tokens_saida']} saída "
              f"({r['tokens_por_s']} tokens/s)")
        if esperadas:
            print(f"Busca trouxe a página esperada {esperadas}? {'SIM' if r['acertou_busca'] else 'NÃO'}")

    a = set(registro["do_zero"]["paginas"])
    b = set(registro["langchain"]["paginas"])
    print(f"\nPáginas em comum: {sorted(a & b) or 'nenhuma'}")
    print(f"Só do zero: {sorted(a - b) or '-'} | Só LangChain: {sorted(b - a) or '-'}")

    os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
    with open(SAIDA, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def garantir_indices():
    if not os.path.exists("vetores.npy"):
        main.indexar(input("Índice do zero não encontrado. Caminho do PDF: "))
    if not os.path.exists(rag_langchain.INDICE):
        caminho = input("Índice LangChain não encontrado. Caminho do PDF: ")
        rag_langchain.indexar(caminho)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("arquivo", nargs="?", help="arquivo de perguntas; sem ele, modo interativo")
    parser.add_argument("--rodadas", type=int, default=1, help="vezes que cada pergunta é feita")
    parser.add_argument("--saida", default=SAIDA, help="arquivo JSONL de resultados")
    args = parser.parse_args()
    SAIDA = args.saida

    garantir_indices()
    if args.arquivo:
        with open(args.arquivo, encoding="utf-8") as f:
            linhas = [linha.strip() for linha in f if linha.strip() and not linha.startswith("#")]
        for rodada in range(1, args.rodadas + 1):
            for i, linha in enumerate(linhas, 1):
                print(f"\n\n########## rodada {rodada}/{args.rodadas} [{i}/{len(linhas)}] {linha}")
                comparar(linha, rodada)
    else:
        while True:
            pergunta = input("\nPergunta (enter vazio para sair): ").strip()
            if not pergunta:
                break
            comparar(pergunta)
    print(f"\nResultados salvos em {SAIDA}")
