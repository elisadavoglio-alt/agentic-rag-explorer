#!/usr/bin/env python3
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
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
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_prompt
    | llm
)

# --- LOOP INTERATTIVO ---
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
