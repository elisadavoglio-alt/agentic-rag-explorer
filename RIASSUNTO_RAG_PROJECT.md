# Progetto Finale: RAG Assistant (Capstone)
**Obiettivo**: Costruire un assistente che risponde alle domande leggendo ESCLUSIVAMENTE da una documentazione personalizzata fornita da noi.

---

## 1. Setup dell'Ambiente
**Obiettivo**: Installare le librerie per il database vettoriale (la memoria a lungo termine).

*   **Comando**:
    ```bash
    venv/bin/pip install chromadb sentence-transformers
    ```
    *(Nota: potrebbe volerci un po', sono librerie pesanti)*

---

## 2. Preparazione dei Dati (Ingestion)
**File**: `ingest.py`

### 🧠 Glossario: Chi sono questi "personaggi"?
*   **Hugging Face** 🤗: È come il "Netflix dell'AI". Un sito dove la gente condivide modelli gratuiti. Noi stiamo scaricando da lì un "cervello" specializzato nel trasformare parole in numeri.
*   **ChromaDB** 🌈: È la "memoria a lungo termine". Salva i **Vettori**.
    *   *Espandono il significato?* **Sì e No**. Non inventano fatti nuovi, ma capiscono i sinonimi. Se nel JSON c'è scritto "auto rossa" e tu chiedi "veicolo scarlatto", il vettore capisce che sono la stessa cosa perché sono vicini nello "spazio dei significati".

*   **Ingest vs Bot**:
    *   `ingest.py` usa un modello piccolo (**Hugging Face**) solo per trasformare testo in numeri.
    *   `rag_bot.py` userà il modello grande (**Llama 3 su Groq**) per leggere quei testi e risponderti come un umano.

### Come funziona lo script:
1.  Legge i tuoi documenti (**JSON** delle pubblicazioni + **Saggi Accademici .txt**).
2.  Li spezzetta in piccoli paragrafi ("Chunks").
3.  Li trasforma in numeri ("Embeddings") e li salva nel database ChromaDB.

---

## 3. L'Assistente (RAG Bot)
**File**: `rag_bot.py`
Questo è l'agente vero e proprio:
1.  Prende la tua domanda.
2.  Cerca nel database i **25 pezzi di testo** più simili (abbiamo aumentato la "memoria" per catturare dettagli tecnici profondi).
3.  Invia tutto a Llama 3 dicendo: "Usa SOLO queste informazioni per rispondere".

---

## 4. Esecuzione
1.  Prima carichiamo i dati: `venv/bin/python 04_RAG_Project/ingest.py`
2.  Poi parliamo col bot: `venv/bin/python 04_RAG_Project/rag_bot.py`

## 5. Interfaccia Grafica (Web UI) ✨
**File**: `app.py`
Abbiamo creato una dashboard web per rendere l’agente accessibile a tutti (non solo agli sviluppatori).

*   **Comando**:
    ```bash
    venv/bin/streamlit run 04_RAG_Project/app.py
    ```
*   **Funzionalità**:
    *   Sidebar per esplorare le pubblicazioni.
    *   Trasparenza: vedi le fonti usate per ogni risposta.
    *   Setup "k" dinamico: puoi decidere quanto far leggere all'agente.
