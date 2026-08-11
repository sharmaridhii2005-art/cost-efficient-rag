# Cost-Efficient RAG

A Retrieval-Augmented Generation (RAG) system designed to answer questions from a document collection while minimizing computational cost, prompt size, and generation overhead.

The project implements document ingestion, chunking, embedding generation, vector storage using ChromaDB, semantic retrieval, local answer generation using FLAN-T5-small, and empirical TOP-K configuration optimization.

---

## 1. Project Objective

The objective of this project is to build a practical and cost-efficient RAG pipeline that can:

* Ingest documents into a searchable knowledge base.
* Split documents into manageable chunks.
* Convert chunks into semantic embeddings.
* Store embeddings in ChromaDB.
* Retrieve the most relevant document chunks for a query.
* Generate grounded answers using a local language model.
* Measure token usage and generation latency.
* Compare different retrieval configurations.
* Select an optimal TOP-K value based on quality, semantic similarity, token usage, latency, and cost.

The project focuses on demonstrating that increasing the number of retrieved chunks does not necessarily improve answer quality and can increase computational overhead.

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
                    TOP_K = 1
                          |
                          v
                 Retrieved Context
                          |
                          v
                 Context Builder
                          |
                          v
                  FLAN-T5-small
                          |
                          v
                    Final Answer
                          |
                          v
              Evaluation & Metrics
```

---

## 3. RAG Pipeline

### Document Ingestion

Documents are loaded from the `documents/` directory using the ingestion module.

```text
documents/
    |
    v
app/ingestion/loader.py
```

### Chunking

Documents are divided into smaller chunks to make semantic retrieval more effective.

Current configuration:

```text
Chunk size   = 500
Chunk overlap = 50
```

Implementation:

```text
app/retrieval/chunker.py
```

### Embedding Generation

Document chunks and user queries are converted into numerical embeddings for semantic similarity search.

Implementation:

```text
app/retrieval/embedder.py
```

### Vector Storage

Embeddings are stored locally using ChromaDB.

```text
chroma_db/
```

Implementation:

```text
app/retrieval/chroma_store.py
```

### Retrieval

The retriever converts the query into an embedding and searches ChromaDB for the most relevant document chunks.

Implementation:

```text
app/retrieval/retriever.py
```

Final configuration:

```text
TOP_K = 1
```

### Local Generation

Retrieved context is passed to a local `google/flan-t5-small` model.

Implementation:

```text
app/generation/local_generator.py
```

The generator is instructed to answer using only the retrieved context.

---

## 4. Project Structure

```text
cost-efficient-rag/
│
├── app/
│   ├── config.py
│   │
│   ├── ingestion/
│   │   └── loader.py
│   │
│   ├── retrieval/
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── retriever.py
│   │   └── chroma_store.py
│   │
│   ├── generation/
│   │   ├── context_builder.py
│   │   ├── generator.py
│   │   ├── local_generator.py
│   │   └── prompt_builder.py
│   │
│   └── api/
│
├── documents/
│
├── chroma_db/
│
├── evaluation/
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
└── README.md
```

---

## 5. Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Sentence Transformers
* ChromaDB
* NumPy
* Pytest
* FLAN-T5-small
* JSON-based evaluation reports

---

## 6. Final Configuration

The final configuration selected through benchmarking is:

| Parameter       |                Value |
| --------------- | -------------------: |
| Chunk Size      |                  500 |
| Chunk Overlap   |                   50 |
| TOP_K           |                    1 |
| Local Model     | google/flan-t5-small |
| Generation Cost |              $0.0000 |

---

## 7. TOP-K Optimization

Two retrieval configurations were evaluated:

* TOP_K = 1
* TOP_K = 3

### Results

| Metric              |        TOP_K=1 |    TOP_K=3 |
| ------------------- | -------------: | ---------: |
| Quality             |      **0.750** |      0.667 |
| Perfect Answers     |        **4/6** |        3/6 |
| Semantic Similarity |     **0.6939** |     0.6721 |
| Avg Prompt Tokens   |     **216.33** |     427.50 |
| Avg Total Tokens    |     **249.50** |     461.17 |
| Generation Latency  |  **973.80 ms** | 1136.48 ms |
| Total Latency       | **1010.88 ms** | 1157.03 ms |
| Cost                |        $0.0000 |    $0.0000 |

### Selected Configuration

```text
TOP_K = 1
```

TOP_K=1 was selected because it achieved higher measured quality and semantic similarity while using substantially fewer tokens and lower latency.

Compared with TOP_K=3:

* Prompt tokens were reduced by approximately 49%.
* Total tokens were reduced by approximately 46%.
* Measured quality increased from 0.667 to 0.750.
* Semantic similarity increased from 0.6721 to 0.6939.
* Total latency was lower.

Therefore, for the current dataset and evaluation set, retrieving one highly relevant chunk provides the best measured cost-efficiency trade-off.

---

## 8. Example Query

### Question

```text
What are the main stages of a RAG system?
```

### Retrieved Context

```text
Source: sample.txt
Chunk: 0
```

The retrieved document describes the main stages as:

```text
document ingestion,
chunking,
embedding generation,
vector storage,
retrieval,
and answer generation
```

### Generated Answer

```text
document ingestion, chunking, embedding generation,
vector storage, retrieval, and answer generation
```

The answer is grounded in the retrieved document context.

---

## 9. Running the Project

Create and activate the virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Index documents:

```powershell
python -m scripts.index_documents
```

Test retrieval:

```powershell
python -m scripts.test_retrieval
```

Test local generation:

```powershell
python -m scripts.test_local_generation
```

Run the end-to-end RAG pipeline:

```powershell
python -m scripts.run_rag
```

Benchmark TOP-K configurations:

```powershell
python -m scripts.benchmark_topk
```

Run configuration comparison:

```powershell
python -m scripts.benchmark_configurations
```

Run tests:

```powershell
pytest -q
```

---

## 10. Evaluation

The project evaluates the RAG pipeline using several measurements:

### Retrieval Evaluation

Measures whether relevant document chunks are retrieved for user queries.

### Generation Quality

Measures whether generated answers correctly answer the evaluation questions.

### Semantic Similarity

Measures semantic similarity between generated answers and expected answers.

### Efficiency

Measures:

* Prompt tokens
* Completion tokens
* Total tokens
* Generation latency
* Total latency
* Generation cost

These measurements allow retrieval configurations to be compared quantitatively.

---

## 11. Test Result

The project test suite currently passes:

```text
1 passed, 1 warning
```

The warning originates from a ChromaDB dependency and does not indicate a failure in the project tests.

---

## 12. Cost Efficiency

The local generation pipeline uses:

```text
google/flan-t5-small
```

Because the model runs locally, the measured API generation cost is:

```text
$0.0000
```

The project additionally reduces unnecessary context by selecting TOP_K=1 based on empirical evaluation.

This reduces the amount of context passed to the generation model and therefore reduces prompt and total token requirements.

---

## 13. Key Findings

The experiments demonstrate that:

1. More retrieved chunks do not automatically produce better answers.
2. TOP_K=1 achieved better measured quality than TOP_K=3.
3. TOP_K=1 achieved higher semantic similarity.
4. TOP_K=1 used significantly fewer prompt tokens.
5. TOP_K=1 had lower generation and total latency.
6. Local generation eliminates per-request API generation cost.
7. Retrieval configuration should be selected using measured quality and efficiency rather than assuming a larger TOP-K is always better.

---

## 14. Limitations

The current evaluation uses a relatively small document collection and evaluation set.

The results therefore demonstrate the behavior of this particular dataset and workload rather than proving that TOP_K=1 is optimal for every RAG application.

For larger production knowledge bases, additional experiments could evaluate:

* Larger TOP_K ranges
* Different chunk sizes
* Different chunk overlaps
* Reranking
* Hybrid search
* Larger local language models
* Quantized models
* GPU inference
* Larger evaluation datasets

---

## 15. Future Improvements

Possible future improvements include:

* Adding a web or API interface.
* Adding streaming generation.
* Adding retrieval confidence thresholds.
* Implementing hybrid keyword + semantic search.
* Adding a reranker.
* Supporting PDF and web document ingestion.
* Adding automated evaluation dashboards.
* Comparing multiple embedding models.
* Comparing multiple local language models.
* Adding model quantization for faster CPU inference.

---

## 16. Conclusion

This project implements a complete Retrieval-Augmented Generation pipeline with local generation and empirical configuration optimization.

The final selected configuration is:

```text
Chunk Size      = 500
Chunk Overlap   = 50
TOP_K           = 1
Model           = google/flan-t5-small
Cost            = $0.0000
```

Based on the current evaluation results, TOP_K=1 provides the best balance between answer quality, semantic similarity, token usage, and latency for this dataset.

The project demonstrates that a cost-efficient RAG system should optimize not only for answer quality, but also for the amount of retrieved context and computational resources required to generate each answer.
