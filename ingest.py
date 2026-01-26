import os
import json
from dotenv import load_dotenv

# Import per il database vettoriale e processing
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# Percorsi
JSON_PATH = "04_RAG_Project/project_1_publications.json"
DB_PATH = "04_RAG_Project/chroma_db"

def ingest_data():
    print("--- INIZIO INGESTION (Caricamento Dati) ---")
    
    # 1. Carica il JSON
    if not os.path.exists(JSON_PATH):
        print(f"ERRORE: Non trovo il file {JSON_PATH}")
        return

    print(f"Leggo {JSON_PATH}...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Trovate {len(data)} pubblicazioni nel JSON.")

    # 2. Converti in Documenti LangChain
    docs = []
    
    # A. Caricamento dal JSON
    for item in data:
        title = item.get("title", "No Title")
        desc = item.get("publication_description", "")
        
        full_text = f"Title: {title}\n\nContent:\n{desc}"
        
        doc = Document(
            page_content=full_text,
            metadata={"title": title, "source": "ReadyTensor"}
        )
        docs.append(doc)
    
    # B. Caricamento del Saggio (Essay) Extra
    essay_path = "04_RAG_Project/rag_systematic_review.txt"
    if os.path.exists(essay_path):
        print(f"Trovato saggio extra: {essay_path}")
        with open(essay_path, 'r', encoding='utf-8') as f:
            essay_text = f.read()
            
        doc_essay = Document(
            page_content=essay_text,
            metadata={"title": "RAG Systems Review 2025", "source": "Academic Essay"}
        )
        docs.append(doc_essay)
    else:
        print("Nessun saggio extra trovato.")
    
    print(f"Preparati {len(docs)} documenti grezzi.")

    # 3. Splitting (Spezzettiamo i testi lunghi)
    # L'AI non può leggere libri intere in un colpo solo, serve dividere in "bocconi" (chunks)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # Ogni pezzo è max 1000 caratteri
        chunk_overlap=200  # Sovrapposizione per non perdere il filo del discorso
    )
    splits = text_splitter.split_documents(docs)
    print(f"Suddivisi in {len(splits)} chunks (pezzetti di testo).")

    # 4. Creazione Database Vettoriale
    print("Salvataggio nel database ChromaDB (questo può richiedere un minuto)...")
    
    # Usiamo un modello di embedding leggero e gratuito di HuggingFace
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Crea e salva il DB su disco
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )
    
    print(f"--- INGESTION COMPLETATA! ---")
    print(f"Database salvato in: {DB_PATH}")

if __name__ == "__main__":
    ingest_data()
