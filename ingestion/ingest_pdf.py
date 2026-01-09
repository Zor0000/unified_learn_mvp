import sys
import os
from typing import List

# --- Path setup ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- LangChain imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# --- Pydantic schema ---
from pydantic import BaseModel

# --- Config ---
from config import POSTGRES_URL


# ================== CONFIG ==================
PDF_PATH = r"D:\RAG_Agent_Internship\data\copilot_studio.pdf"

NEW_COLLECTION = "copilot_pdf_agentic"

MAX_PAGES = None           # ✅ process only first 100 pages
MAX_CHARS_PER_PAGE = 4000 # ✅ safe context limit per LLM call
# ============================================
# High-level retrieval classification
TAG_MAP = {
    "definition": "concept",
    "explanation": "concept",
    "steps": "how_to",
    "example": "reference"
}


# ================== SCHEMA ==================
class AgenticChunk(BaseModel):
    chunk_title: str
    chunk_text: str
    chunk_summary: str
    chunk_type: str          # definition | steps | explanation | example
    keywords: List[str]
    difficulty: str          # beginner | intermediate | advanced
# ============================================
class AgenticChunkList(BaseModel):
    chunks: List[AgenticChunk]


# ================== PROMPT ==================
AGENTIC_CHUNK_PROMPT = """
Split the following technical documentation into semantically meaningful
chunks suitable for a Retrieval-Augmented Generation (RAG) system.

Rules:
- One clear concept per chunk
- Do not exceed ~400 words per chunk
- Do not split steps or explanations mid-way

Return the result as a list under the key `chunks`.

Text:
<<<
{TEXT}
>>>
"""

# ============================================


def agentic_chunk(text_block: str, llm) -> List[AgenticChunk]:
    result = llm.invoke([
        HumanMessage(
            content=AGENTIC_CHUNK_PROMPT.format(
                TEXT=text_block[:MAX_CHARS_PER_PAGE]
            )
        )
    ])
    return result.chunks


def agentic_chunks_to_documents(
    chunks: List[AgenticChunk],
    source: str,
    page_number: int
) -> List[Document]:
    docs = []

    for c in chunks:
        tag = TAG_MAP.get(c.chunk_type, "general")

        docs.append(
            Document(
                page_content=c.chunk_text,
                metadata={
                    "title": c.chunk_title,
                    "summary": c.chunk_summary,
                    "chunk_type": c.chunk_type,
                    "tag": tag,                      
                    "keywords": c.keywords,
                    "difficulty": c.difficulty,
                    "page_number": page_number,      
                    "source": source
                }
            )
        )

    return docs



def ingest_pdf_agentic():
    print("📄 Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"✅ Loaded {len(documents)} pages total")

    
    if MAX_PAGES:
        documents = documents[:MAX_PAGES]

    print(f"📉 Processing first {len(documents)} pages only")

    print("🧠 Initializing LLM with structured output...")
    llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
    ).with_structured_output(AgenticChunkList)


    all_docs: List[Document] = []

    print("✂️ Performing agentic chunking...")
    for idx, doc in enumerate(documents, start=1):
        try:
            chunks = agentic_chunk(doc.page_content, llm)
            docs = agentic_chunks_to_documents(
                chunks,
                source=PDF_PATH,
                page_number=idx
            )
            all_docs.extend(docs)
            print(f"  ✔ Page {idx}: {len(docs)} agentic chunks")
        except Exception as e:
            print(f"  ⚠️ Page {idx} failed:", e)

    print(f"📦 Total chunks to embed: {len(all_docs)}")

    if not all_docs:
        print("❌ No chunks generated. Exiting.")
        return

    print("🔢 Creating embeddings...")
    embeddings = OpenAIEmbeddings()

    print("🗄️ Storing in NEW pgvector collection...")
    db = PGVector(
        connection_string=POSTGRES_URL,
        embedding_function=embeddings,
        collection_name=NEW_COLLECTION
    )

    db.add_documents(all_docs)

    print("✅ Agentic ingestion complete!")
    print(f"📚 Collection used: {NEW_COLLECTION}")
    print("🛡️ Old embeddings remain untouched.")


if __name__ == "__main__":
    ingest_pdf_agentic()
