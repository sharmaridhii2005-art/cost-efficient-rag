# Cost-Efficient RAG

A Retrieval-Augmented Generation (RAG) system designed to answer questions from a document collection while minimizing computational cost, prompt size, and generation overhead.

The project implements an end-to-end RAG pipeline consisting of:

* Document ingestion
* Text chunking
* Embedding generation
* ChromaDB vector storage
* Semantic retrieval
* Retrieval distance filtering
* Context building
* Local answer generation using FLAN-T5-small
* Token and latency measurement
* TOP-K benchmarking
* Semantic evaluation
* Generation quality evaluation
* Configuration comparison
* Automated testing

The project focuses on finding a practical retrieval configuration that provides good answer quality while avoiding unnecessary context and computational overhead.

---

## 1. Project Objective

The objective of this project is to build a practical and cost-efficient RAG pipeline that can:

* Ingest documents into a searchable knowledge base.
* Split documents into manageable chunks.
* Convert chunks into semantic embeddings.
* Store embeddings in ChromaDB.
* Retrieve relevant document chunks for a query.
* Reject weak retrieval results using a distance threshold.
* Build context from retrieved documents.
* Generate grounded answers using a local language model.
* Measure token usage and generation latency.
* Compare different TOP-K configurations.
* Evaluate generated answer quality.
* Measure semantic similarity.
* Select a suitable TOP-K configuration based on empirical results.

The project demonstrates that retrieving more document chunks does not automatically improve answer quality and can increase prompt size and computational overhead.

---

## 2. System Architecture

```text
                         User Question
                              |
                              v
                     Query Embedding
                              |
                              v
                         ChromaDB
                       Vector Search
                              |
                              v
                    Retrieval Distance
                         Filtering
                              |
                              v
                         TOP_K = 2
                              |
                              v
                   Retrieved Documents
                              |
                              v
                     Context Builder
                              |
                              v
                     Prompt Builder
                              |
                              v
                    FLAN-T5-small
                              |
                              v
                       Final Answer
                              |
                              v
                 Token / Latency Metrics
                              |
                              v
                 Evaluation & Benchmarking
```

---

## 3. RAG Pipeline

### 3.1 Document Ingestion

Documents are loaded from the `documents/` directory.

Implementation:

```text
app/ingestion/loader.py
```

Example document collection:

```text
documents/
├── rag_guide.txt
└── sample.txt
```

The loader reads the documents and creates structured document objects containing information such as:

* Source
* Text
* Document metadata

---

### 3.2 Chunking

Documents are divided into smaller chunks to improve semantic retrieval.

Current configuration:

```text
Chunk Size   = 500
Chunk Overlap = 50
```

Implementation:

```text
app/retrieval/chunker.py
```

Chunking prevents very large documents from being passed to the embedding and generation stages as a single unit.

---

### 3.3 Embedding Generation

Each document chunk is converted into a numerical vector representation.

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Implementation:

```text
app/retrieval/embedder.py
```

The same embedding model is used to convert the user's query into a vector so that semantic similarity can be calculated between the query and stored document chunks.

---

### 3.4 Vector Storage

Document embeddings are stored locally using ChromaDB.

Storage directory:

```text
chroma_db/
```

Implementation:

```text
app/retrieval/chroma_store.py
```

ChromaDB provides persistent vector storage and similarity search.

---

### 3.5 Retrieval

The retriever performs the following operations:

```text
User Query
    |
    v
Query Embedding
    |
    v
ChromaDB Similarity Search
    |
    v
Retrieve TOP_K chunks
    |
    v
Distance Threshold Filtering
    |
    v
Relevant Context
```

Implementation:

```text
app/retrieval/retriever.py
```

The current retrieval configuration is:

```text
TOP_K = 2
```

A retrieval distance threshold is also applied:

```text
RETRIEVAL_DISTANCE_THRESHOLD = 1.2
```

Higher ChromaDB distances indicate weaker semantic matches.

If the retrieved documents exceed the configured distance threshold, the system can reject them instead of passing weak context to the generation model.

This provides a safer fallback for questions that are not supported by the document collection.

---

## 4. Context Building

Retrieved document chunks are converted into a structured context before generation.

Implementation:

```text
app/generation/context_builder.py
```

The context contains information such as:

```text
Source
Chunk Index
Document Text
```

This allows the generation stage to use the retrieved evidence rather than relying only on the language model's internal knowledge.

---

## 5. Prompt Building

The retrieved context and user question are combined into a prompt.

Implementation:

```text
app/generation/prompt_builder.py
```

The generation model is instructed to answer using the retrieved context.

This helps keep the generated answer grounded in the document collection.

---

## 6. Local Generation

The project uses the local Hugging Face model:

```text
google/flan-t5-small
```

Implementation:

```text
app/generation/local_generator.py
```

The model runs locally instead of using a paid external generation API.

The generation stage records:

* Prompt tokens
* Completion tokens
* Total tokens
* Generation latency
* Estimated cost

Because generation is performed locally, the measured API generation cost is:

```text
$0.0000
```

---

## 7. Retrieval Safety / Fallback

The system includes a retrieval distance threshold.

Current value:

```text
RETRIEVAL_DISTANCE_THRESHOLD = 1.2
```

For example, a question such as:

```text
Who is the President of the United States?
```

does not have supporting information in the current project documents.

Instead of generating an unsupported answer, the system returns:

```text
I don't have enough information in the provided documents.
```

This prevents the RAG system from treating an unrelated retrieved document as valid evidence.

---

## 8. Project Structure

```text
cost-efficient-rag/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── loader.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── retriever.py
│   │   └── chroma_store.py
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── context_builder.py
│   │   ├── generator.py
│   │   ├── local_generator.py
│   │   └── prompt_builder.py
│   │
│   └── api/
│       └── __init__.py
│
├── documents/
│   ├── rag_guide.txt
│   └── sample.txt
│
├── chroma_db/
│
├── evaluation/
│   ├── data/
│   │   ├── generation_questions.json
│   │   └── retrieval_questions.json
│   │
│   ├── retrieval_evaluator.py
│   │
│   └── results/
│       ├── retrieval_baseline.json
│       ├── generation_topk_baseline.json
│       ├── generation_topk_evaluation.json
│       ├── generation_semantic_evaluation.json
│       └── configuration_benchmark/
│           └── configuration_comparison.json
│
├── reports/
│
├── tests/
│   ├── test_chunker.py
│   ├── test_context_builder.py
│   ├── test_loader.py
│   ├── test_prompt_builder.py
│   └── test_retriever.py
│
├── scripts/
│   ├── index_documents.py
│   ├── test_retrieval.py
│   ├── test_local_generation.py
│   ├── benchmark_topk.py
│   ├── benchmark_configurations.py
│   ├── evaluate_generation.py
│   ├── evaluate_semantic.py
│   └── run_rag.py
│
├── .env
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 9. Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Sentence Transformers
* ChromaDB
* NumPy
* Pytest
* FLAN-T5-small
* JSON
* Git
* GitHub

---

## 10. Final Configuration

The current configuration selected after benchmarking is:

| Parameter                    |                                    Value |
| ---------------------------- | ---------------------------------------: |
| Embedding Model              | `sentence-transformers/all-MiniLM-L6-v2` |
| Chunk Size                   |                                      500 |
| Chunk Overlap                |                                       50 |
| TOP_K                        |                                    **2** |
| Retrieval Distance Threshold |                                      1.2 |
| Local Model                  |                   `google/flan-t5-small` |
| Generation Cost              |                                  $0.0000 |

---

# 11. TOP-K Optimization

The project evaluates different numbers of retrieved chunks to determine whether increasing TOP-K improves answer quality.

The benchmark was performed using:

```text
TOP_K = 1
TOP_K = 2
TOP_K = 3
```

The generation benchmark produced the following averages:

| Metric                 |   TOP_K=1 |       TOP_K=2 |   TOP_K=3 |
| ---------------------- | --------: | ------------: | --------: |
| Avg Prompt Tokens      |    216.33 |        307.50 |    373.17 |
| Avg Total Tokens       |    249.50 |        338.83 |    406.83 |
| Avg Generation Latency | 949.77 ms | **717.79 ms** | 731.45 ms |

The results demonstrate that increasing TOP-K substantially increases the amount of context supplied to the generation model.

---

## 12. Generation Quality Evaluation

The generation quality evaluation produced:

| TOP_K   |   Quality | Perfect Answers |
| ------- | --------: | --------------: |
| TOP_K=1 | **0.750** |         **4/6** |
| TOP_K=2 | **0.750** |         **4/6** |
| TOP_K=3 |     0.667 |             3/6 |

The important observation is that TOP_K=2 maintains the same measured quality as TOP_K=1 while allowing additional relevant context to be retrieved.

TOP_K=3 reduced measured quality to:

```text
0.667
```

with only:

```text
3/6 perfect answers
```

---

## 13. Semantic Generation Evaluation

Semantic similarity evaluation produced:

| TOP_K   | Average Similarity | Minimum | Maximum |
| ------- | -----------------: | ------: | ------: |
| TOP_K=1 |             0.6939 |  0.3144 |  0.8929 |
| TOP_K=2 |         **0.6970** |  0.2068 |  0.8929 |
| TOP_K=3 |             0.6919 |  0.2068 |  0.8929 |

TOP_K=2 achieved the highest average semantic similarity:

```text
0.6970
```

compared with:

```text
TOP_K=1 → 0.6939
TOP_K=3 → 0.6919
```

Therefore, the semantic evaluation also supports TOP_K=2 as a strong configuration for the current dataset.

---

## 14. Selected TOP-K Configuration

The final configuration is:

```text
TOP_K = 2
```

The selection is based on the combined experimental results.

TOP_K=2:

* Maintains the highest measured generation quality level of 0.750.
* Produces 4/6 perfect answers.
* Achieves the highest average semantic similarity of 0.6970.
* Provides more retrieval context than TOP_K=1.
* Avoids the quality decrease observed with TOP_K=3.
* Provides a practical balance between context and generation overhead.

The important conclusion is not that TOP_K=2 is universally optimal.

Instead:

> TOP_K should be selected empirically based on the target dataset, retrieval quality, answer quality, token usage, latency, and cost.

---

# 15. Example Query

### Question

```text
What are the main stages of a RAG system?
```

### Retrieved Context

```text
Source: sample.txt
Chunk: 0
```

The retrieved document states that the main stages are:

```text
document ingestion,
chunking,
embedding generation,
vector storage,
retrieval,
and answer generation
```

Another retrieved chunk from `rag_guide.txt` provides additional information about retrieval and answer generation.

### Generated Answer

```text
document ingestion, chunking, embedding generation,
vector storage, retrieval, and answer generation
```

The answer is grounded in the retrieved document context.

---

# 16. Example of Retrieval Rejection

### Question

```text
Who is the President of the United States?
```

The current document collection does not contain information answering this question.

The retrieval threshold prevents weakly related context from being treated as valid evidence.

The system returns:

```text
I don't have enough information in the provided documents.
```

This demonstrates the fallback behavior of the RAG system.

---

# 17. Running the Project

### Create the virtual environment

```powershell
python -m venv venv
```

### Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

---

## 18. Index Documents

Run:

```powershell
python -m scripts.index_documents
```

This loads the documents, creates chunks, generates embeddings, and stores them in ChromaDB.

---

## 19. Test Retrieval

Run:

```powershell
python -m scripts.test_retrieval
```

---

## 20. Test Local Generation

Run:

```powershell
python -m scripts.test_local_generation
```

---

## 21. Run the Complete RAG Pipeline

Run:

```powershell
python -m scripts.run_rag
```

Example:

```text
======================================================================
COST-EFFICIENT RAG - END-TO-END PIPELINE
======================================================================

Enter your question: What are the main stages of a RAG system?
```

The system performs:

```text
Question
   ↓
Embedding
   ↓
ChromaDB Retrieval
   ↓
Distance Filtering
   ↓
Context Building
   ↓
Prompt Construction
   ↓
FLAN-T5-small
   ↓
Answer
   ↓
Metrics
```

---

# 22. Benchmark TOP-K

Run:

```powershell
python -m scripts.benchmark_topk
```

The benchmark compares:

```text
TOP_K = 1
TOP_K = 2
TOP_K = 3
```

Results are saved under:

```text
evaluation/results/
```

---

# 23. Run Generation Quality Evaluation

Run:

```powershell
python -m scripts.evaluate_generation
```

The current evaluation produced:

```text
TOP_K=1 | Quality=0.75 | Perfect=4/6
TOP_K=2 | Quality=0.75 | Perfect=4/6
TOP_K=3 | Quality=0.667 | Perfect=3/6
```

---

# 24. Run Semantic Evaluation

Run:

```powershell
python -m scripts.evaluate_semantic
```

Current results:

```text
TOP_K=1 | Average similarity=0.6939
TOP_K=2 | Average similarity=0.6970
TOP_K=3 | Average similarity=0.6919
```

---

# 25. Run Configuration Benchmark

Run:

```powershell
python -m scripts.benchmark_configurations
```

The resulting comparison is stored in:

```text
evaluation/results/configuration_benchmark/configuration_comparison.json
```

---

# 26. Run Automated Tests

Run:

```powershell
pytest -q
```

Current result:

```text
15 passed, 1 warning
```

The warning is a `DeprecationWarning` originating from the ChromaDB/OpenTelemetry dependency:

```text
asyncio.iscoroutinefunction
```

It does not indicate a failure in the project's tests.

---

# 27. Evaluation Results

The generated evaluation files are stored in:

```text
evaluation/results/
```

Current tracked results include:

```text
evaluation/results/retrieval_baseline.json

evaluation/results/generation_topk_baseline.json

evaluation/results/generation_topk_evaluation.json

evaluation/results/generation_semantic_evaluation.json

evaluation/results/configuration_benchmark/
└── configuration_comparison.json
```

These files provide reproducible evidence for the configuration decisions made in the project.

---

# 28. Cost Efficiency

The generation model is:

```text
google/flan-t5-small
```

and runs locally.

Therefore, the measured API generation cost is:

```text
$0.0000
```

Cost efficiency is additionally improved by controlling the amount of retrieved context.

Increasing TOP_K increases the number of chunks passed toward generation and can therefore increase prompt size and computational requirements.

The experiments demonstrate that more context does not automatically produce better answers.

---

# 29. Key Findings

The experiments demonstrate that:

1. More retrieved chunks do not automatically produce better answers.
2. TOP_K=3 produced lower generation quality than TOP_K=1 and TOP_K=2.
3. TOP_K=2 achieved the highest measured semantic similarity.
4. TOP_K=1 and TOP_K=2 both achieved a quality score of 0.750.
5. TOP_K=1 and TOP_K=2 both produced 4/6 perfect answers.
6. TOP_K=2 provides additional context while maintaining the highest measured quality level.
7. Increasing TOP_K increases prompt and total token usage.
8. Local generation eliminates API generation cost.
9. Retrieval distance filtering provides a safer fallback for unsupported questions.
10. Retrieval configuration should be selected using measured quality and efficiency rather than assuming that a larger TOP_K is always better.

---

# 30. Limitations

The current evaluation uses a relatively small document collection and evaluation dataset.

Therefore, the results demonstrate the behavior of this particular dataset and workload.

They do not prove that TOP_K=2 is optimal for every RAG application.

Other datasets may require a different configuration.

The current system also uses a relatively small local language model:

```text
google/flan-t5-small
```

which may produce shorter or less detailed answers than larger language models.

---

# 31. Future Improvements

Possible future improvements include:

* Adding a web or API interface.
* Adding streaming generation.
* Implementing hybrid keyword + semantic search.
* Adding a reranker.
* Supporting PDF document ingestion.
* Supporting web document ingestion.
* Adding automated evaluation dashboards.
* Comparing multiple embedding models.
* Comparing multiple local language models.
* Adding model quantization.
* Improving CPU inference speed.
* Expanding the evaluation dataset.
* Testing larger TOP_K ranges.
* Benchmarking different chunk sizes.
* Benchmarking different chunk overlaps.
* Adding retrieval precision and recall metrics.
* Adding citation-aware answers.
* Adding answer confidence scores.

---

# 32. Conclusion

This project implements a complete Retrieval-Augmented Generation pipeline using semantic embeddings, persistent ChromaDB storage, retrieval filtering, context construction, and local FLAN-T5-small generation.

The current selected configuration is:

```text
Embedding Model = sentence-transformers/all-MiniLM-L6-v2
Chunk Size      = 500
Chunk Overlap   = 50
TOP_K           = 2
Distance Limit  = 1.2
Model           = google/flan-t5-small
Generation Cost = $0.0000
```

The experimental evaluation found:

```text
TOP_K=1 → Quality 0.750 | Semantic Similarity 0.6939
TOP_K=2 → Quality 0.750 | Semantic Similarity 0.6970
TOP_K=3 → Quality 0.667 | Semantic Similarity 0.6919
```

Based on these measurements, TOP_K=2 is currently selected because it maintains the best measured generation quality level while achieving the highest semantic similarity among the tested configurations.

The project demonstrates an important principle of cost-efficient RAG:

> The goal is not to retrieve the maximum amount of information. The goal is to retrieve enough relevant information to produce a high-quality grounded answer while minimizing unnecessary computation and context.

For the current dataset, **TOP_K=2 provides the best measured balance between retrieval context, answer quality, semantic similarity, and efficiency.**
