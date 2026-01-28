import script_fix_sqlite
import streamlit as st
import os
import json
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# Carica variabili d'ambiente
load_dotenv()

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="RAG Explorer Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS per look & feel moderno
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stChatMessage.user {
        background-color: #f0f2f6;
    }
    .stChatMessage.assistant {
        background-color: #e8f5e9;
        border: 1px solid #c8e6c9;
    }
    h1 {
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Titolo Principale
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712009.png", width=80)
with col2:
    st.title("Ready Tensor & Academic RAG Explorer")
    st.caption("Semantic search engine powered by Llama 3 and ChromaDB")

# Percorsi Robusti (Cloud vs Local)
# Se siamo in cloud, la cartella 04_RAG_Project non esiste come sottocartella, siamo già dentro.
BASE_DIR = "04_RAG_Project" if os.path.exists("04_RAG_Project") else "."
JSON_PATH = os.path.join(BASE_DIR, "project_1_publications.json")
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

ESSAY_PATH = os.path.join(BASE_DIR, "rag_systematic_review.txt")

# --- SIDEBAR: DOCUMENT EXPLORER ---
with st.sidebar:
    st.header("📂 Document Library")
    
    # 1. Preparazione Lista Titoli
    titles = []
    data = []
    
    # Carica JSON
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, 'r') as f:
            data = json.load(f)
        titles = [item['title'] for item in data]
        titles.sort()
    
    # Aggiungi voce manuale per il Saggio .txt
    essay_title = "📄 RAG Systems Review 2025 (Full Text)"
    if os.path.exists(ESSAY_PATH):
        titles.insert(0, essay_title) # Mettilo in cima

    # 2. Dropdown
    if titles:
        selected_title = st.selectbox("Select a publication:", ["(Select...)"] + titles)
        
        # 3. Visualizzazione Contenuto
        if selected_title != "(Select...)":
            st.subheader(selected_title)
            
            # Caso A: È il saggio TXT
            if selected_title == essay_title:
                st.info("**Type**: Academic Systematic Review (Text File source)")
                with open(ESSAY_PATH, 'r', encoding='utf-8') as f:
                    essay_content = f.read()
                st.text_area("Content Preview", essay_content, height=400)
                
            # Caso B: È un articolo dal JSON
            else:
                selected_item = next((item for item in data if item['title'] == selected_title), None)
                if selected_item:
                    st.info(f"**ID**: {selected_item.get('id', 'N/A')}")
                    st.markdown(f"**Abstract**:\n{selected_item.get('publication_description', 'No description')}")
    else:
        st.warning("No documents found (JSON missing?).")

    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    k_slider = st.slider("Retrieval Depth (k)", min_value=1, max_value=50, value=25)
    st.caption("Number of chunks retrieved per query.")

# --- MOTORE RAG ---

@st.cache_resource
def get_rag_chain():
    if not os.path.exists(DB_PATH):
        st.warning("⚠️ Database vettoriale non trovato. Avvio Ingestion automatica (Cloud Mode)...")
        # Importiamo e lanciamo l'ingestion al volo
        from ingest import ingest_data
        with st.spinner("⏳ Sto leggendo i documenti e creando il database... (Richiederà 1-2 minuti)"):
            ingest_data()
        st.success("✅ Database creato con successo! Ricarica la pagina se necessario.")
        st.rerun()

    # 1. Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Vector Store
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    
    # Ritorniamo il vectorstore per poter cambiare k dinamicamente
    return vectorstore

vectorstore = get_rag_chain()

if vectorstore:
    # Aggiorna il retriever in base allo slider
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_slider})
    
    # LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    
    # Prompt
    template = """You are an expert academic research assistant.
    Answer the question using ONLY the provided context.
    Be precise, cite facts, and maintain a professional tone.
    If you don't know the answer, state "I don't know based on the provided documents".

    Context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(f"[Source: {doc.metadata.get('title', 'Unknown')}] {doc.page_content}" for doc in docs)

    # Catena Avanzata: Ritorna anche le fonti
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    # --- INTERFACCIA CHAT ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask about papers or agentic AI..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("🧠 Analyzing documents..."):
                try:
                    # Eseguiamo la chain che ritorna dizionario {answer, context}
                    result = rag_chain_with_source.invoke(query)
                    answer = result["answer"]
                    sources = result["context"]

                    st.markdown(answer)
                    
                    # Mostra Fonti in Expander
                    with st.expander("📚 Retrieved Sources (RAG Transparency)"):
                        for i, doc in enumerate(sources[:5]): # Mostra solo le prime 5
                            st.markdown(f"**Source {i+1}:** {doc.metadata.get('title', 'N/A')}")
                            st.caption(f"{doc.page_content[:200]}...")
                            st.divider()
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
