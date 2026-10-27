"""Mesmo RAG do main.py, mas montado com as peças prontas do LangChain."""
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from main import EMBED_MODEL, LLM, PROMPT

INDICE = "indice_langchain.json"

embeddings = OllamaEmbeddings(model=EMBED_MODEL)
llm = ChatOllama(model=LLM, temperature=0.2, reasoning=False)
chain = ChatPromptTemplate.from_template(PROMPT) | llm | StrOutputParser()
_store = None


def indexar(caminho_pdf):
    paginas = PyPDFLoader(caminho_pdf).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(paginas)
    store = InMemoryVectorStore.from_documents(chunks, embeddings)
    store.dump(INDICE)
    print(f"{len(chunks)} chunks indexados (LangChain).")


def carregar():
    global _store
    if _store is None:
        _store = InMemoryVectorStore.load(INDICE, embeddings)
    return _store


def buscar(pergunta, k=5):
    docs = carregar().similarity_search_with_score(pergunta, k=k)
    # mesmo formato do main.buscar: ({"pagina", "texto"}, score)
    return [({"pagina": d.metadata["page"] + 1, "texto": d.page_content}, float(s))
            for d, s in docs]


def gerar_resposta(pergunta):
    resultados = buscar(pergunta)
    contexto = "\n\n".join(f"[página {c['pagina']}]\n{c['texto']}" for c, _ in resultados)
    texto = chain.invoke({"contexto": contexto, "pergunta": pergunta})
    return texto, resultados


if __name__ == "__main__":
    comando = input("Gostaria de indexar ou perguntar algo? ( 1 para indexar, 2 para perguntar): ")
    if comando == "1":
        indexar(input("Digite o caminho do PDF: "))
    elif comando == "2":
        texto, resultados = gerar_resposta(input("Qual é a sua pergunta? "))
        print(texto)
        print("\nFontes:", sorted({c["pagina"] for c, _ in resultados}))
    else:
        print("Comando inválido. Escolha '1' ou '2'.")
