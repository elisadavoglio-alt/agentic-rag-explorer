#!/usr/bin/env python3
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Carica variabili d'ambiente (es. .env)
load_dotenv()

# --- CONFIGURAZIONE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
API_KEY = os.environ.get("GROQ_API_KEY")

if not API_KEY:
    print("⚠️ ERRORE: La variabile d'impiego GROQ_API_KEY non è settata!")
    exit(1)

# --- CARICAMENTO DATABASE ---
print("⏳ Caricamento Database Vettoriale...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if not os.path.exists(DB_PATH):
    print("❌ Database non trovato! Esegui prima 'python ingest.py'")
    exit(1)

vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
# Impostiamo k=25 come nella app web per coerenza
retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

# --- LLM SETUP ---
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# --- PROMPT ---
template = """
You are an expert assistant. Answer using ONLY the provided context. 
If the answer is not in the documents, state: "Non lo so in base ai documenti forniti".

Context:
{context}

Question:
{question}

Answer:
"""
custom_prompt = PromptTemplate.from_template(template)

# --- CATENA RAG ---
# --- QUERY EXPANSION & RAG CHAIN ---
print("⚙️  Configurazione Query Expansion...")

# 1. Generatore di Varianti (Multi-Query)
query_gen_template = """
You are an AI language model assistant. Your task is to generate 3 different versions 
of the given user question to retrieve relevant documents from a vector database. 
By generating multiple perspectives on the user question, your goal is to help 
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines.

Original question: {question}
"""
query_gen_prompt = PromptTemplate.from_template(query_gen_template)

# Chain per generare le query: (Question) -> [Query1, Query2, Query3]
generate_queries_chain = (
    query_gen_prompt 
    | llm 
    | (lambda x: x.content.split("\n"))
)

def get_unique_union_docs(question: str):
    """
    1. Genera 3 varianti della domanda.
    2. Per ogni variante, recupera i documenti.
    3. Unisce e rimuove i duplicati.
    """
    print(f"   🔍 Generating variations for: '{question}'")
    queries = generate_queries_chain.invoke({"question": question})
    
    # Pulizia query vuote
    queries = [q.strip() for q in queries if q.strip()]
    if not queries: 
        queries = [question] # Fallback
        
    print(f"   💡 Variations: {queries}")
    
    # Recupero per ogni query
    all_docs = []
    for q in queries:
        docs = retriever.invoke(q)
        all_docs.extend(docs)
    
    # Deduplicazione (basata sul contenuto o ID se disponibile)
    unique_docs = []
    seen_content = set()
    for doc in all_docs:
        if doc.page_content not in seen_content:
            unique_docs.append(doc)
            seen_content.add(doc.page_content)
            
    print(f"   📚 Retrieved {len(unique_docs)} unique docs total.")
    return unique_docs

# 2. Catena Finale
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Modifichiamo la chain per usare la funzione custom di retrieval
# Avvolgiamo le funzioni in RunnableLambda per usare la pipe |
rag_chain = (
    {"context": RunnableLambda(get_unique_union_docs) | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | custom_prompt
    | llm
)

# --- LOOP INTERATTIVO ---
if __name__ == "__main__":
    print("\n🤖 RAG Bot Initilized (Terminal Mode)")
    print("Scrivi la tua domanda (o 'exit' per uscire):\n")

    while True:
        query = input("User: ")
        if query.lower() in ["exit", "quit", "q"]:
            print("Bye! 👋")
            break
        
        if not query.strip():
            continue

        print("Agent: Thinking...")
        try:
            response = rag_chain.invoke(query)
            print(f"Agent: {response.content}\n")
            print("-" * 50)
        except Exception as e:
            print(f"❌ Errore: {e}")
