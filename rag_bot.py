import os
from dotenv import load_dotenv

# Import LangChain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Percorsi
DB_PATH = "04_RAG_Project/chroma_db"

def format_docs(docs):
    """Formatta i documenti trovati in un unico blocco di testo."""
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    print("--- RAG AGENT DIRETTORE (Caricamento...) ---")
    
    # 1. Carichiamo la Memoria (Il Database creato da ingest.py)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if not os.path.exists(DB_PATH):
        print("ERRORE: Database non trovato. Hai lanciato 'ingest.py'?")
        return

    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    
    # 2. Creiamo il "Retriever" (Il Cercatore)
    # k=25 per assicurarsi di trovare i dettagli nel saggio (che è molto ricco)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

    # 3. Il Cervello (Llama 3)
    llm = ChatGroq(model="llama-3.3-70b-versatile")

    # 4. Il Prompt (Le istruzioni per l'AI)
    template = """Sei un assistente esperto. Rispondi alla domanda usando SOLO il contesto fornito qui sotto.
    Se non trovi la risposta nel contesto, dì onestamente "Non lo so in base ai documenti forniti".

    Contesto:
    {context}

    Domanda: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 5. La Catena (The Chain) 🔗
    # Questa è la "ricetta" di LangChain:
    # Prendi domanda -> Cerca documenti -> Uniscili -> Manda al Prompt -> Manda all'LLM -> Leggi risposta
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("--- AGENTE PRONTO! Fai domande sulle pubblicazioni Ready Tensor ---")
    print("(Scrivi 'esci' per chiudere)")

    while True:
        query = input("\nTu: ")
        if query.lower() in ["esci", "exit"]:
            print("Ciao!")
            break
        
        print("🤖 Ragiono sui documenti...")
        try:
            # Eseguiamo la catena
            response = rag_chain.invoke(query)
            print(f"Agente: {response}")
        except Exception as e:
            print(f"Errore: {e}")

if __name__ == "__main__":
    main()
