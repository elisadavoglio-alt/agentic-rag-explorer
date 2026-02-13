import os
import sys
from rag_bot import get_unique_union_docs, retriever

# --- CONFIGURAZIONE DATASET DI TEST (GROUND TRUTH) ---
# Domanda -> Stringa che DEVE essere presente nel documento recuperato
TEST_DATASET = [
    {
        "question": "What is the difference between RAG-Sequence and RAG-Token?",
        "expected_content_snippet": "RAG-Sequence", 
        "criteria": "Must mention RAG-Sequence definition"
    },
    {
        "question": "How does RAG-Token work?",
        "expected_content_snippet": "switching documents per token", # Keyword dal paper di Lewis
        "criteria": "Must describe per-token retrieval"
    },
    {
        "question": "List agentic frameworks discussed.",
        "expected_content_snippet": "ReadyTensor", # Assumendo che i json abbiano questa fonte
        "criteria": "Must retrieve ReadyTensor publications"
    }
]

def evaluate():
    print("📊 AVVIO VALUTAZIONE RETRIEVAL (Hit Rate & MRR)")
    print(f"   Test Set Size: {len(TEST_DATASET)} questions")
    print("-" * 50)

    hits = 0
    mrr_sum = 0

    for i, item in enumerate(TEST_DATASET):
        q = item["question"]
        expected = item["expected_content_snippet"]
        
        print(f"\n[{i+1}/{len(TEST_DATASET)}] Question: {q}")
        
        # Eseguiamo la retrieval (con Query Expansion)
        # Nota: get_unique_union_docs restituisce una lista di Documenti
        retrieved_docs = get_unique_union_docs(q)
        
        found_rank = -1
        
        for rank, doc in enumerate(retrieved_docs):
            # Controllo se il contenuto atteso è nel chunk
            # (Case insensitive per robustezza)
            if expected.lower() in doc.page_content.lower():
                found_rank = rank + 1 # Rank 1-based
                break
        
        if found_rank > 0:
            print(f"   ✅ HIT! Found at rank #{found_rank}")
            hits += 1
            mrr_sum += (1.0 / found_rank)
        else:
            print(f"   ❌ MISS. Expected snippet '{expected}' not found in top docs.")

    # --- CALCOLO METRICHE ---
    hit_rate = hits / len(TEST_DATASET)
    mrr = mrr_sum / len(TEST_DATASET)

    print("\n" + "=" * 50)
    print("📈 RISULTATI FINALI")
    print("=" * 50)
    print(f"🎯 Hit Rate: {hit_rate:.2%}")
    print(f"🏆 MRR (Mean Reciprocal Rank): {mrr:.4f}")
    print("=" * 50)
    
    if hit_rate < 0.7:
        print("⚠️  Warning: Retrieval performance is low. Consider increasing k or improving chunks.")
    else:
        print("✅ System is performing well (Review Ready!)")

if __name__ == "__main__":
    evaluate()
