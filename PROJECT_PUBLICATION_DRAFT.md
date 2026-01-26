# Beyond Fixed Knowledge: An Agentic RAG Framework

## 📑 Abstract
This project addresses the limitation of static Large Language Models (LLMs) by developing a specialized **Retrieval-Augmented Generation (RAG)** assistant. Unlike generic chatbots, this system is engineered to ingest and sympathize knowledge from two distinct sources—structured industry data (Ready Tensor Publications) and unstructured academic research (Systematic Reviews). By optimizing retrieval depth (`k=25`) and leveraging state-of-the-art vector embeddings, the assistant delivers precise, verifiable answers on niche technical inputs.

---

## 1. Project Motivation
General-purpose models like GPT-4 or Llama 3 excel at broad reasoning but hallucinate when asked about private datasets or very recent specialized research. 
Our goal was to build a **"Domain Sentinel"**: an AI that refuses to guess and instead relies exclusively on a curated "Second Brain". We specifically targeted the challenge of **Multi-Source Ingestion**, requiring the system to unify clean JSON metadata with dense academic text into a single queryable vector space.

## 2. System Architecture
The solution is built on a modular pipeline using **LangChain** for orchestration and **ChromaDB** for persistent memory.

```mermaid
graph TD
    subgraph "Knowledge Ingestion"
        A[JSON Data] -->|Parse| C(Text Chunks)
        B[Academic Essays .txt] -->|Read| C
        C -->|Embed (Hugging Face)| D[(ChromaDB Vector Store)]
    end
    
    subgraph "Query Processing"
        Q[User Question] -->|Embed| V(Query Vector)
        V -->|Semantic Search (k=25)| D
        D -->|Retrieve Context| P{System Prompt}
        P -->|Generate Answer| LLM[Llama 3-70b]
    end
```

## 3. The "Multi-Source" Engine
Most RAG tutorials focus on loading simple PDFs. We architected a **Hybrid Ingestion Module** (`ingest.py`) capable of synthesizing heterogeneous data:

*   **Structured Data**: JSON publications are parsed to exact fields (`title`, `description`).
3.  **Unstructured Knowledge**: Academic essays (e.g., `rag_systematic_review.txt`) are ingested raw.

The system uses `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=200) to ensure context continuity across both data types. We confirmed that `sentence-transformers/all-MiniLM-L6-v2` offers the best balance of speed and semantic capture for this mixed dataset.

This allows the agent to answer questions like *"What is the difference between RAG-Sequence and RAG-Token?"*—a nuance often found only in deep academic literature, not in high-level summaries.

## 4. Optimization: The "Deep Recall" Strategy
During testing, we encountered the **"Lost in the Middle" phenomenon**.
*   *Initial State*: With `k=3` (retrieving the top 3 chunks), the agent frequently answered "I don't know" to specific technical queries.
*   *Diagnosis*: Semantic search placed the exact definitions of technical terms around rank #15-#20.
*   *Optimization*: We increased the retrieval horizon to **`k=25`** in `rag_bot.py`.

```python
# rag_bot.py optimized configuration
retriever = vectorstore.as_retriever(search_kwargs={"k": 25})
```
This adjustment significantly boosted the agent's ability to "connect the dots" between scattered pieces of information without overwhelming the LLM's context window.

## 5. Prompt Engineering: From "Generalist" to "Specialist"
A critical phase was refining the system instructions. We moved from a generic prompt to a strict "Sentinel" persona.

**❌ Initial Attempt (Too Permissive)**:
> "Answer the question based on the context. If you don't know, try to be helpful."
> *Result*: The model would hallucinate answers using its pre-trained knowledge, ignoring our private data.

**✅ Final "Sentinel" Prompt**:
> "You are an expert assistant. Answer using **ONLY** the provided context. If the answer is not in the documents, state: 'Non lo so in base ai documenti forniti'."
> *Result*: Zero hallucinations. The model now acts as a strict gateway to the verified knowledge base.

## 6. Performance & Results
The system demonstrates high fidelity in retrieval.

> **Test Case 1: specialized concepts**
> *User*: "What is the difference between RAG-Sequence and RAG-Token?"
> *Agent*: Correctly identifies RAG-Sequence as generating the full response based on one document, while RAG-Token allows switching documents per token (citing Lewis et al., 2020).

> **Test Case 2: Content Discovery**
> *User*: "What frameworks are discussed for agentic AI?"
> *Agent*: Retrieves Ready Tensor publication abstracts regarding specialized agent frameworks.

## 6. Interactive Web Interface
To make the system accessible beyond the command line, we developed a modern web dashboard using **Streamlit** (`app.py`).

**Key Features**:
*   **Document Explorer**: A sidebar allows users to browse and read the raw content of ingested publications and essays.
*   **Hyper-Parameter Tuning**: A real-time slider to adjust the retrieval depth (`k`) from 1 to 50, strictly for advanced analysis.
*   **Source Transparency**: Every answer includes an expandable "Retrieved Sources" section, showing the exact text chunks used by the agent to generate the response.

## 7. Deployment Guide
To replicate this "Second Brain":

1.  **Environment**: `pip install -r requirements.txt` (requires `langchain`, `chromadb`, `groq`).
2.  **Ingestion**: Run `python ingest.py` to build the vector store from `project_1_publications.json` and `rag_systematic_review.txt`.
3.  **Interaction**:
    *   **CLI Mode**: `python rag_bot.py`
    *   **Web UI**: `streamlit run app.py`

## 7. Future Directions
The next evolution of this project will involve **GraphRAG**—constructing a knowledge graph to explicitly model relationships between authors, concepts, and papers, further reducing the reliance on pure semantic proximity.

## 8. License & Attribution
This project is released under the **MIT License**, promoting open innovation and academic sharing.

### Third-Party Licenses
*   **ChromaDB**: Apache License 2.0
*   **HuggingFace Embeddings**: Apache License 2.0
*   **LangChain**: MIT License
*   **Streamlit**: Apache License 2.0

## 9. Acknowledgments
This Agentic RAG Framework was designed and implemented by **Elisa Davoglio**.

Special thanks to:
*   **Ready Tensor** for the foundational dataset and high-quality educational resources.
*   The open-source community behind LangChain and Llama 3 for providing the building blocks of modern AI agents.
