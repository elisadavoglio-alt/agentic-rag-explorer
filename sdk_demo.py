import os
from openai import OpenAI
from dotenv import load_dotenv

# 1. Carichiamo la chiave (come al solito)
load_dotenv()

# 2. Creiamo il CLIENT (il "telecomando")
# Nota: Usiamo la libreria 'openai', ma puntiamo ai server di Groq!
# Questo dimostra che l'SDK è "universale" come diceva la lezione.
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# 3. Testo da riassumere (Simuliamo un'email ricevuta)
email_text = """
Da: Mario Rossi
A: Team
Oggetto: Aggiornamento Progetto Alpha

Ciao a tutti,
volevo aggiornarvi sul fatto che abbiamo riscontrato dei ritardi nella consegna del modulo B.
Il fornitore ha avuto problemi di logistica per via dello sciopero. 
Per questo motivo, la riunione di domani è cancellata.
Vi prego di inviarmi i vostri report aggiornati entro venerdì sera invece che giovedì.
Grazie,
Mario
"""

print(f"📧 EMAIL ORIGINALE:\n{email_text}")
print("-" * 40)
print("🤖 L'AI sta leggendo e riassumendo (senza LangChain)...")

# 4. Chiamata DIRETTA (Bare Metal)
# Qui parliamo direttamente al modello. Niente "PromptTemplate", niente "Chains".
# Solo: "Tieni questo messaggio e dammi una risposta".
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile", # Usiamo il modello veloce di Groq
    messages=[
        # Il "System" dà l'istruzione generale (il cervello)
        {"role": "system", "content": "Sei un assistente efficiente. Riassumi il testo in una singola frase, evidenziando le scadenze."},
        # L'"User" dà i dati (l'email)
        {"role": "user", "content": email_text}
    ],
    temperature=0.0 # Vogliamo precisione massima
)

# 5. Estrazione Risultato
summary = response.choices[0].message.content
print("-" * 40)
print(f"📝 RIASSUNTO:\n{summary}")
