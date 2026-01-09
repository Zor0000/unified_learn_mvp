from sqlalchemy import create_engine

POSTGRES_URL = "postgresql+psycopg2://postgres:neeraj@localhost:5433/ms_rag_db"

engine = create_engine(POSTGRES_URL)

try:
    conn = engine.connect()
    print("✅ Postgres connected successfully")
    conn.close()
except Exception as e:
    print("❌ Connection failed")
    print(e)
