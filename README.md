# RAG do zero vs. LangChain: conversando com o manual da moto

Um assistente que responde dúvidas sobre o manual de uma Yamaha Fazer FZ25 ABS 2023, rodando 100% local com [Ollama](https://ollama.com).

O mesmo pipeline foi implementado duas vezes:

- **Do zero** ([main.py](main.py)): pypdf, NumPy e chamadas HTTP diretas ao Ollama. Sem framework e sem banco vetorial.
- **Com LangChain** ([rag_langchain.py](rag_langchain.py)): loader, splitter, vector store e chain prontos do framework.

O objetivo é um estudo comparativo: o que o framework facilita, o que esconde e se muda a qualidade das respostas.

## Como funciona

```
PDF → texto por página → chunks → embeddings (bge-m3) → índice
                                                          ↓
pergunta → embedding → similaridade de cosseno → top-5 chunks → prompt → qwen3.5:4b → resposta + páginas
```

| Etapa | Do zero | LangChain |
|---|---|---|
| Leitura do PDF | `pypdf.PdfReader` | `PyPDFLoader` |
| Chunking | janela fixa de 800 caracteres, sobreposição de 150, sem cruzar páginas | `RecursiveCharacterTextSplitter` (800/150), corta preferindo parágrafos e frases |
| Embeddings | `POST /api/embed` em lotes de 32 | `OllamaEmbeddings` |
| Índice | `vetores.npy` + `chunks.json` | `InMemoryVectorStore` salvo em `indice_langchain.json` |
| Busca | produto escalar de vetores normalizados com NumPy | `similarity_search_with_score` |
| Geração | `POST /api/chat` | `ChatPromptTemplate \| ChatOllama \| StrOutputParser` |

## Estrutura

```
main.py              RAG do zero (indexação, busca e resposta) + CLI
rag_langchain.py     o mesmo RAG com LangChain + CLI
comparar.py          faz a mesma pergunta às duas versões e registra os resultados
app.py               interface web (Flask) com o chat ao lado do PDF
static/index.html    front-end: chat + visualizador pdf.js com as páginas citadas
estudo/perguntas.txt perguntas do estudo com as páginas corretas (gabarito)
resultados/          saída do comparar.py (comparacao.jsonl)
files/               coloque o PDF do manual aqui (não versionado)
```

## Como rodar

Pré-requisitos: Python 3.9+ e Ollama rodando.

```bash
ollama pull bge-m3
ollama pull qwen3.5:4b
python -m venv venv
venv/bin/pip install -r requirements.txt
```

Baixe o manual da Fazer FZ25 ABS 2023 no site da Yamaha e salve como `files/manual_fazerfz25abs_2023.pdf`. Para usar outro manual, troque `PDF_NAME` em [app.py](app.py).

Indexe nas duas versões:

```bash
venv/bin/python main.py                # opção 1, caminho do PDF
venv/bin/python rag_langchain.py       # opção 1, caminho do PDF
```

Use:

```bash
venv/bin/python app.py                 # interface em http://localhost:5050
venv/bin/python comparar.py            # pergunta às duas versões, modo interativo
venv/bin/python comparar.py estudo/perguntas.txt   # roda o estudo inteiro
```

## O estudo

### Variáveis controladas

Para que a diferença venha só da forma de ler, quebrar e buscar o texto, as duas versões compartilham:

- modelo de embedding: `bge-m3`
- LLM: `qwen3.5:4b`, temperatura 0.2, sem modo de raciocínio
- o mesmo prompt (`PROMPT` em [main.py](main.py))
- tamanho de chunk (800) e sobreposição (150)
- top-k = 5

### Conjunto de perguntas

[estudo/perguntas.txt](estudo/perguntas.txt) tem uma pergunta por linha, com as páginas do PDF que contêm a resposta depois do `|`:

```
Qual a folga correta da corrente? | 71
```

As páginas são as do arquivo PDF, não a numeração impressa no manual (a página 64 do PDF é a "6-18" do manual).

### Métricas

Registradas por pergunta e por versão em `resultados/comparacao.jsonl`:

| Métrica | Como é medida |
|---|---|
| Acerto da busca (recall@5) | alguma página do gabarito apareceu entre os 5 chunks recuperados (`acertou_busca`) |
| Acerto da resposta | avaliação manual: a resposta está correta e completa? |
| Fidelidade | avaliação manual: a resposta inventou algo que não está nos trechos? |
| Tempo de busca e total | `tempo_busca_s` e `tempo_total_s` |
| Concordância | páginas recuperadas em comum entre as duas versões |

Separar o acerto da busca do acerto da resposta mostra onde está o erro: se a página certa foi recuperada e a resposta saiu errada, o problema é do LLM, não do RAG.

Para resumir o arquivo de resultados:

```bash
venv/bin/python -c "
import json
linhas = [json.loads(l) for l in open('resultados/comparacao.jsonl')]
for v in ('do_zero', 'langchain'):
    com_gabarito = [l[v] for l in linhas if 'acertou_busca' in l[v]]
    acertos = sum(r['acertou_busca'] for r in com_gabarito)
    tempo = sum(r['tempo_total_s'] for r in com_gabarito) / len(com_gabarito)
    print(f'{v:10} recall@5 {acertos}/{len(com_gabarito)}  tempo médio {tempo:.1f}s')
"
```

### Primeiros achados (8 perguntas)

- **Busca:** as duas versões trouxeram a página certa nas 8 perguntas, e quase sempre as mesmas páginas.
- **"Pressão dos pneus com garupa?"**: as duas recuperaram a página 64, que diz "2 pessoas: 200/225 kPa", mas responderam que não há informação sobre garupa. A busca acertou; o modelo de 4B não ligou "garupa" a "2 pessoas".
- **"Quantos litros de óleo vão na troca?"**: o trecho certo foi recuperado, mas a versão do zero trocou os rótulos ("1,35 L com remoção do filtro", quando 1,35 L é sem o filtro).
- **O próprio manual se contradiz:** a especificação de óleo diz API "SL ou superior" na página 96 e "SG ou superior" na página 89.

### Próximos passos

- [ ] Ampliar o gabarito para 20+ perguntas, incluindo perguntas com sinônimos ("garupa", "calibrar") e perguntas sem resposta no manual
- [ ] Juntar palavras hifenizadas pela quebra de linha ("aciden- tes") antes do chunking
- [ ] Busca híbrida (BM25 + vetor) para termos exatos como códigos de peça
- [ ] Mostrar a numeração impressa do manual nas citações
- [ ] Testar outros tamanhos de chunk e um LLM maior
