# Ready Tensor RAG Assistant 🤖📚

This project is a **Retrieval-Augmented Generation (RAG)** assistant built to explore Ready Tensor publications. It allows users to ask natural language questions and receive accurate answers based *solely* on the provided documentation, avoiding AI hallucinations.

## 🚀 Features
- **Data Ingestion**: Parses JSON publications and converts them into vector embeddings.
- **Vector Database**: Uses **ChromaDB** for efficient similarity search.
- **RAG Pipeline**: Combines **LangChain**, **Hugging Face Embeddings**, and **Llama 3** (via Groq) to generate context-aware responses.
- **Source Citation**: The answers are grounded in the specific articles provided.

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **Orchestration**: LangChain
- **LLM**: Llama 3 (via Groq API)
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Database**: ChromaDB

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd 04_RAG_Project
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables**
   Create a `.env` file and add your Groq API Key:
   ```
   GROQ_API_KEY=gsk_...
   ```

## 🚦 Usage

### Step 1: Ingest Data
Before chatting, you must load the data into the vector database.
```bash
python ingest.py
```
*Expected Output:* `--- INGESTION COMPLETATA! ---`

### Step 2: Run the Bot
Start the interactive chat session.
```bash
python rag_bot.py
```

### Example Interaction
```text
Tu: What is UV package manager?
Agente: UV is a Python package manager built in Rust that is 10-100x faster than pip...
```

## 📂 Project Structure
- `ingest.py`: Loads `project_1_publications.json`, chunks text, and saves to ChromaDB.
- `rag_bot.py`: Main application script that retrieves data and generates answers.
- `chroma_db/`: Directory containing the vector database (generated after ingestion).
