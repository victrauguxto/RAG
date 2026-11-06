# RAG do zero vs. LangChain: avaliando um assistente para o manual da moto

Um assistente que responde dúvidas sobre o manual de uma Yamaha Fazer FZ25 ABS 2023, rodando 100% local com [Ollama](https://ollama.com), e um estudo que avalia esse assistente com métricas.

O mesmo pipeline de RAG (Retrieval-Augmented Generation) foi implementado duas vezes:

- **Do zero** ([main.py](main.py)): pypdf, NumPy e chamadas HTTP diretas ao Ollama. Sem framework e sem banco vetorial.
- **Com LangChain** ([rag_langchain.py](rag_langchain.py)): loader, splitter, vector store e chain prontos do framework.

As duas versões foram comparadas com um conjunto de perguntas com gabarito, separando a qualidade da busca, a qualidade da resposta e o custo de cada uma.

## Principais resultados

- As duas versões ficaram praticamente empatadas: **27/39** respostas corretas no RAG do zero e **28/39** com LangChain, em 3 rodadas de 13 perguntas.
- O LangChain usou **~10% mais tokens** por pergunta, porque o splitter dele gera trechos maiores.
- A busca foi a mesma nas duas versões em quase todas as perguntas. As diferenças vieram do chunking e do modelo, não do framework.
- A maior parte dos erros não está no modelo: está na extração do PDF e na busca, antes de o LLM receber qualquer coisa.

## Arquitetura

```
PDF → texto por página → chunks → embeddings (bge-m3) → índice
                                                          ↓
pergunta → embedding → similaridade de cosseno → top-5 chunks → prompt → qwen3.5:4b → resposta + páginas
```

| Etapa | Do zero | LangChain |
|---|---|---|
| Leitura do PDF | `pypdf.PdfReader` | `PyPDFLoader` (também usa pypdf) |
| Chunking | janela fixa de 800 caracteres, sobreposição de 150, sem cruzar páginas | `RecursiveCharacterTextSplitter` (800/150), corta preferindo parágrafos e linhas |
| Embeddings | `POST /api/embed` em lotes de 32 | `OllamaEmbeddings` |
| Índice | `vetores.npy` + `chunks.json` | `InMemoryVectorStore` salvo em `indice_langchain.json` |
| Busca | produto escalar de vetores normalizados (NumPy) | `similarity_search_with_score` (cosseno com NumPy) |
| Geração | `POST /api/chat` | `ChatPromptTemplate \| ChatOllama` |

A busca usa só a função densa do bge-m3 (um vetor por trecho), que é o que o Ollama expõe.

## Metodologia de avaliação

### Variáveis controladas

As duas versões compartilham:

- modelo de embedding: `bge-m3`
- LLM: `qwen3.5:4b`, temperatura 0.2, sem modo de raciocínio
- o mesmo prompt (`PROMPT` em [main.py](main.py))
- tamanho de chunk (800), sobreposição (150) e top-k (5)

A única variável é a implementação.

### Gabarito

[estudo/perguntas.txt](estudo/perguntas.txt) tem 13 perguntas. Para cada uma, foram anotadas à mão as páginas do PDF que contêm a resposta e os fatos que uma resposta correta precisa ter:

```
Qual a folga correta da corrente? | 71 | 40,0; 45,0
```

As páginas são as do arquivo PDF, não a numeração impressa no manual (a página 64 do PDF é a "6-18" do manual).

### Métricas

| Camada | Métrica | O que mede |
|---|---|---|
| Busca | hit@5 por página | alguma página do gabarito apareceu entre os 5 trechos recuperados |
| Busca | MRR por página | quão no topo veio o primeiro trecho de uma página certa (1 = primeiro lugar) |
| Busca | hit@5 por trecho | algum trecho recuperado contém de fato os fatos da resposta |
| Geração | acerto por fatos-chave | a resposta contém todos os fatos do gabarito |
| Geração | revisão manual | rótulos trocados, citação de página errada e respostas parciais |
| Geração | consistência | a pergunta teve o mesmo veredito nas 3 rodadas |
| Operação | latência, tokens de entrada e de saída, tokens/s | custo e velocidade de cada versão |

Busca e geração são avaliadas separadamente. Assim dá para saber se um erro aconteceu porque o trecho certo não foi encontrado ou porque o modelo errou mesmo com o trecho certo em mãos.

## Resultados

Cada uma das 13 perguntas foi feita 3 vezes a cada versão, totalizando 78 respostas. Os dados estão em [resultados/estudo_3_rodadas.jsonl](resultados/estudo_3_rodadas.jsonl) e as tabelas abaixo saem do [avaliar.py](avaliar.py).

### Busca

| Métrica | Do zero | LangChain |
|---|---|---|
| hit@5 (página) | 12/13 | 11/13 |
| MRR (página) | 0,87 | 0,76 |
| hit@5 (trecho) | 11/13 | 11/13 |

O hit@5 por página favorece o RAG do zero, mas engana: em "Quanto de óleo deve ser usado no motor?", ele trouxe a página certa e o trecho errado dela. Medido por trecho, o resultado empata.

### Geração e operação

| Métrica | Do zero | LangChain |
|---|---|---|
| Acerto por fatos-chave (automático) | 30/39 (77%) | 32/39 (82%) |
| **Respostas corretas após revisão manual** | **27/39 (69%)** | **28/39 (72%)** |
| Consistência entre rodadas | 13/13 perguntas | 12/13 perguntas |
| Latência média | 3,9 s | 4,3 s |
| Tokens de entrada por pergunta | 1.107 | 1.217 |
| Tokens de saída por pergunta | 89 | 102 |
| Velocidade | 27,7 tokens/s | 27,4 tokens/s |

A checagem por fatos-chave não detecta tudo. A revisão manual tirou estas respostas da conta:

| Pergunta | Versão | Rodadas | Problema |
|---|---|---|---|
| Quantos litros de óleo vão na troca? | do zero | 3/3 | Trocou os rótulos: disse que 1,35 L é "com remoção do filtro" (é sem) |
| Qual a carga máxima que a moto suporta? | LangChain | 2/3 | Valor certo, mas citou a "página 3", que vem de um rodapé do PDF |
| Qual a pressão dos pneus com garupa? | LangChain | 2/3 | Listou os valores de 1 e 2 pessoas, mas respondeu que não encontrou nada sobre garupa |

### Análise de erros por etapa do pipeline

| Falha | Etapa | Causa |
|---|---|---|
| Citação "página 3" | Extração | O rodapé do arquivo original ("UBC5W4W0.book Page 3...") entra no texto extraído |
| Bateria "não encontrada" (as duas versões, 3/3) | Busca | A especificação (modelo XTZ7L, 12 V, 6 Ah) está numa tabela técnica da página 90 que não ficou entre os 5 trechos mais similares |
| "Quanto de óleo deve ser usado?" (as duas, 3/3) | Chunking e busca | A quantidade fica num trecho que a busca não recuperou. Com a pergunta "Quantos litros de óleo vão na troca?", o trecho certo é encontrado |
| Garupa não associada a "2 pessoas" | Geração | O trecho certo foi recuperado e o modelo não fez a ligação |
| Litros de óleo com rótulos trocados (do zero) | Chunking e geração | O chunking do zero junta a lista numa linha só. O LangChain preserva as quebras de linha e acertou as 3 vezes |

### Conclusões

1. **O framework não foi o fator decisivo.** O `InMemoryVectorStore` faz a mesma busca do NumPy. A diferença veio de um padrão do LangChain, o splitter que preserva a estrutura do texto, que ajudou em uma pergunta e custou ~10% mais tokens.
2. **"Não encontrei" pode ser falso.** Quando a busca falha, o modelo afirma com confiança que a informação não existe no manual.
3. **A métrica precisa ter a granularidade certa.** Avaliar a busca por página escondeu um erro que só apareceu ao avaliar por trecho.
4. **A forma de perguntar muda o resultado.** Duas perguntas com o mesmo sentido tiveram resultados opostos.

### Limitações

- **Amostra pequena.** 13 perguntas não permitem afirmar que 28/39 é melhor que 27/39 com significância estatística.
- **Um único avaliador** fez o gabarito e a revisão manual.
- **Sem perguntas fora do manual.** O estudo não mede se o sistema se recusa a responder quando deveria.
- **A fidelidade aos trechos não foi medida automaticamente** (por exemplo, com LLM-as-judge ou RAGAS).

## Como rodar

Pré-requisitos: Python 3.9+ e [Ollama](https://ollama.com) rodando.

```bash
ollama pull bge-m3
ollama pull qwen3.5:4b
python -m venv venv
venv/bin/pip install -r requirements.txt
```

Baixe o manual da Fazer FZ25 ABS 2023 no site da Yamaha e salve como `files/manual_fazerfz25abs_2023.pdf`. O PDF não é versionado porque tem direitos autorais da fabricante. Para usar outro manual, troque `PDF_NAME` em [app.py](app.py) e refaça o gabarito.

Indexe nas duas versões (opção 1 do menu, informando o caminho do PDF):

```bash
venv/bin/python main.py
venv/bin/python rag_langchain.py
```

Use:

```bash
venv/bin/python app.py              # interface web em http://localhost:5050
venv/bin/python comparar.py         # faz cada pergunta às duas versões, modo interativo
```

Reproduza o estudo:

```bash
venv/bin/python comparar.py estudo/perguntas.txt --rodadas 3 --saida resultados/estudo_3_rodadas.jsonl
venv/bin/python avaliar.py resultados/estudo_3_rodadas.jsonl
```

## Estrutura

```
main.py              RAG do zero: extração, chunking, embeddings, busca e geração + CLI
rag_langchain.py     o mesmo RAG com LangChain + CLI
comparar.py          faz as mesmas perguntas às duas versões e grava os resultados em JSONL
avaliar.py           calcula as métricas de busca, geração e operação
app.py               interface web (Flask) com o chat ao lado do PDF
static/index.html    front-end: chat + visualizador pdf.js com as páginas citadas
estudo/perguntas.txt gabarito: perguntas, páginas esperadas e fatos-chave
resultados/          resultados das rodadas (JSONL)
```

## Próximos passos

- [ ] Ampliar o gabarito para 50+ perguntas, incluindo sinônimos e perguntas sem resposta no manual
- [ ] Limpar o texto extraído: remover rodapés e juntar palavras hifenizadas ("aciden- tes")
- [ ] Testar as funções esparsa e multivetorial do bge-m3 (busca híbrida), focando em termos exatos como códigos de peça
- [ ] Medir fidelidade com LLM-as-judge
- [ ] Transformar a busca em uma ferramenta usada por um agente (LangGraph)
