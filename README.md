# RAG from scratch vs. LangChain: evaluating an assistant for a motorcycle manual

An assistant that answers questions about the owner's manual of a Yamaha Fazer FZ25 ABS 2023, running 100% locally with [Ollama](https://ollama.com), and a study that evaluates it with metrics.

The same RAG (Retrieval-Augmented Generation) pipeline was implemented twice:

- **From scratch** ([main.py](main.py)): pypdf, NumPy and direct HTTP calls to Ollama. No framework and no vector database.
- **With LangChain** ([rag_langchain.py](rag_langchain.py)): the framework's ready-made loader, splitter, vector store and chain.

Both versions were compared on a hand-labeled question set, evaluating retrieval quality, answer quality and cost separately.

> The manual, the questions and the answers are in Portuguese. Questions quoted below come with an English translation.

## Key results

- Both versions were practically tied: **27/39** correct answers from scratch and **28/39** with LangChain, over 3 runs of 13 questions.
- LangChain used **~10% more tokens** per question, because its splitter produces larger chunks.
- Retrieval was the same in both versions for almost every question. The differences came from chunking and the model, not the framework.
- Most errors are not in the model: they happen in PDF extraction and retrieval, before the LLM receives anything.

## Architecture

```
PDF → text per page → chunks → embeddings (bge-m3) → index
                                                        ↓
question → embedding → cosine similarity → top-5 chunks → prompt → qwen3.5:4b → answer + pages
```

| Step | From scratch | LangChain |
|---|---|---|
| PDF reading | `pypdf.PdfReader` | `PyPDFLoader` (also uses pypdf) |
| Chunking | fixed 800-character window, 150 overlap, never crossing pages | `RecursiveCharacterTextSplitter` (800/150), prefers cutting at paragraphs and lines |
| Embeddings | `POST /api/embed` in batches of 32 | `OllamaEmbeddings` |
| Index | `vetores.npy` + `chunks.json` | `InMemoryVectorStore` saved to `indice_langchain.json` |
| Retrieval | dot product of normalized vectors (NumPy) | `similarity_search_with_score` (cosine with NumPy) |
| Generation | `POST /api/chat` | `ChatPromptTemplate \| ChatOllama` |

Retrieval uses only bge-m3's dense function (one vector per chunk), which is what Ollama exposes.

## Evaluation methodology

### Controlled variables

Both versions share:

- embedding model: `bge-m3`
- LLM: `qwen3.5:4b`, temperature 0.2, no reasoning mode
- the same prompt (`PROMPT` in [main.py](main.py))
- chunk size (800), overlap (150) and top-k (5)

The only variable is the implementation.

### Golden set

[estudo/perguntas.txt](estudo/perguntas.txt) has 13 questions. For each one, the PDF pages that contain the answer and the facts a correct answer must include were labeled by hand:

```
Qual a folga correta da corrente? | 71 | 40,0; 45,0
```

(*"What is the correct chain slack?" | expected page | required facts*)

Pages are the PDF file's pages, not the numbering printed in the manual (PDF page 64 is the manual's "6-18").

### Metrics

| Layer | Metric | What it measures |
|---|---|---|
| Retrieval | hit@5 by page | some golden-set page appeared among the 5 retrieved chunks |
| Retrieval | MRR by page | how close to the top the first chunk from a correct page came (1 = first place) |
| Retrieval | hit@5 by chunk | some retrieved chunk actually contains the facts of the answer |
| Generation | key-fact accuracy | the answer contains all the golden-set facts |
| Generation | manual review | swapped labels, wrong page citations and partial answers |
| Generation | consistency | the question got the same verdict in all 3 runs |
| Operations | latency, input and output tokens, tokens/s | cost and speed of each version |

Retrieval and generation are evaluated separately. That shows whether an error happened because the right chunk wasn't found or because the model got it wrong even with the right chunk in hand.

## Results

Each of the 13 questions was asked 3 times to each version, for 78 answers in total. The data is in [resultados/estudo_3_rodadas.jsonl](resultados/estudo_3_rodadas.jsonl), and the tables below come from [avaliar.py](avaliar.py).

### Retrieval

| Metric | From scratch | LangChain |
|---|---|---|
| hit@5 (page) | 12/13 | 11/13 |
| MRR (page) | 0.87 | 0.76 |
| hit@5 (chunk) | 11/13 | 11/13 |

hit@5 by page favors the from-scratch version, but it's misleading: for "How much oil should be used in the engine?", it retrieved the right page but the wrong chunk from it. Measured by chunk, the result is a tie.

### Generation and operations

| Metric | From scratch | LangChain |
|---|---|---|
| Key-fact accuracy (automatic) | 30/39 (77%) | 32/39 (82%) |
| **Correct answers after manual review** | **27/39 (69%)** | **28/39 (72%)** |
| Consistency across runs | 13/13 questions | 12/13 questions |
| Mean latency | 3.9 s | 4.3 s |
| Input tokens per question | 1,107 | 1,217 |
| Output tokens per question | 89 | 102 |
| Speed | 27.7 tokens/s | 27.4 tokens/s |

The key-fact check doesn't catch everything. Manual review removed these answers from the count:

| Question | Version | Runs | Problem |
|---|---|---|---|
| "How many liters of oil go in an oil change?" | from scratch | 3/3 | Swapped the labels: said 1.35 L is "with filter removal" (it's without) |
| "What is the maximum load the motorcycle supports?" | LangChain | 2/3 | Right value, but cited "page 3", which comes from a PDF footer |
| "What is the tire pressure with a passenger?" | LangChain | 2/3 | Listed the values for 1 and 2 riders, but answered that it found nothing about a passenger |

### Error analysis by pipeline stage

| Failure | Stage | Cause |
|---|---|---|
| "Page 3" citation | Extraction | The original file's footer ("UBC5W4W0.book Page 3...") ends up in the extracted text |
| Battery "not found" (both versions, 3/3) | Retrieval | The spec (model XTZ7L, 12 V, 6 Ah) is in a technical table on page 90 that wasn't among the 5 most similar chunks |
| "How much oil should be used?" (both, 3/3) | Chunking and retrieval | The quantity is in a chunk retrieval didn't return. With the question "How many liters of oil go in an oil change?", the right chunk is found |
| Passenger not linked to "2 riders" | Generation | The right chunk was retrieved and the model didn't make the connection |
| Oil liters with swapped labels (from scratch) | Chunking and generation | From-scratch chunking merges the list into a single line. LangChain keeps the line breaks and got it right all 3 times |

### Conclusions

1. **The framework wasn't the deciding factor.** `InMemoryVectorStore` does the same search as NumPy. The difference came from a LangChain default, the splitter that preserves text structure, which helped on one question and cost ~10% more tokens.
2. **"Not found" can be false.** When retrieval fails, the model confidently states that the information isn't in the manual.
3. **A metric needs the right granularity.** Evaluating retrieval by page hid an error that only showed up when evaluating by chunk.
4. **How you ask changes the result.** Two questions with the same meaning got opposite results.

### Limitations

- **Small sample.** 13 questions don't allow claiming that 28/39 beats 27/39 with statistical significance.
- **A single reviewer** built the golden set and did the manual review.
- **No out-of-scope questions.** The study doesn't measure whether the system refuses to answer when it should.
- **Faithfulness to the retrieved chunks wasn't measured automatically** (for example, with LLM-as-judge or RAGAS).

## How to run

Prerequisites: Python 3.9+ and [Ollama](https://ollama.com) running.

```bash
ollama pull bge-m3
ollama pull qwen3.5:4b
python -m venv venv
venv/bin/pip install -r requirements.txt
```

Download the Fazer FZ25 ABS 2023 manual from Yamaha's website and save it as `files/manual_fazerfz25abs_2023.pdf`. The PDF isn't versioned because it's copyrighted by the manufacturer. To use another manual, change `PDF_NAME` in [app.py](app.py) and rebuild the golden set.

Index with both versions (menu option 1, giving the PDF path):

```bash
venv/bin/python main.py
venv/bin/python rag_langchain.py
```

Use:

```bash
venv/bin/python app.py              # web interface at http://localhost:5050
venv/bin/python comparar.py         # asks each question to both versions, interactive mode
```

Reproduce the study:

```bash
venv/bin/python comparar.py estudo/perguntas.txt --rodadas 3 --saida resultados/estudo_3_rodadas.jsonl
venv/bin/python avaliar.py resultados/estudo_3_rodadas.jsonl
```

## Structure

```
main.py              from-scratch RAG: extraction, chunking, embeddings, retrieval and generation + CLI
rag_langchain.py     the same RAG with LangChain + CLI
comparar.py          asks both versions the same questions and writes the results to JSONL
avaliar.py           computes the retrieval, generation and operations metrics
app.py               web interface (Flask) with the chat next to the PDF
static/index.html    front end: chat + pdf.js viewer showing the cited pages
estudo/perguntas.txt golden set: questions, expected pages and key facts
resultados/          results of the runs (JSONL)
```

## Next steps

- [ ] Expand the golden set to 50+ questions, including synonyms and questions with no answer in the manual
- [ ] Clean the extracted text: remove footers and join hyphenated words ("aciden- tes")
- [ ] Test bge-m3's sparse and multi-vector functions (hybrid search), focusing on exact terms such as part codes
- [ ] Measure faithfulness with LLM-as-judge
- [ ] Turn retrieval into a tool used by an agent (LangGraph)
