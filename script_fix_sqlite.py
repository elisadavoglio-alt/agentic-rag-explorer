import sys
import sqlite3

# Tenta di forzare l'uso di pysqlite3 (necessario per server Linux vecchi come Streamlit Cloud)
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("✅ [Fix] Usato 'pysqlite3' (Fix per Cloud attivo).")
except ModuleNotFoundError:
    # Se fallisce (es. su Mac o se non installato), usa quello di sistema che di solito va bene
    print(f"⚠️ [Fix] 'pysqlite3' non trovato. Uso sqlite3 di sistema (v{sqlite3.sqlite_version}).")
    pass
