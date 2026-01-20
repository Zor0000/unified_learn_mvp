# FAST Copilot Studio PDF Ingestion (Milvus-based + Batched + Cached)

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- Path setup ----------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------- PDF ----------
import fitz  # PyMuPDF

# ---------- LangChain ----------
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# ---------- Pydantic ----------
from pydantic import BaseModel, Field

# ============================================================
# CONFIG
# ============================================================

PDF_PATH = r"D:\RAG_Agent_Internship\data\copilot_studio.pdf"
COLLECTION_NAME = "rag_chunks"

MAX_WORKERS = 4
H2_BATCH_SIZE = 3

CACHE_DIR = Path(".llm_cache")
CACHE_DIR.mkdir(exist_ok=True)

# ============================================================
# CONTROLLED TOPIC TAXONOMY
# ============================================================

COPILOT_TOPICS = [
    "agent_overview",
    "agent_architecture",
    "agent_orchestration",
    "generative_ai_flows",
    "agent_authoring",
    "conversation_design",
    "prompt_engineering",
    "connectors_and_plugins",
    "external_apis",
    "data_sources",
    "authentication_and_security",
    "environment_management",
    "deployment_and_publishing",
    "testing_and_debugging",
    "monitoring_and_analytics",
    "governance_and_compliance",
    "limits_and_quotas",
    "support_and_community",
    "agent_management",
    "troubleshooting",
    "content_moderation",
]

# ============================================================
# H2 DETECTION
# ============================================================

def is_h2(text: str, font_size: float) -> bool:
    if font_size < 15:
        return False
    if len(text) > 120:
        return False
    if text.endswith((".", ":", ";", ",")):
        return False
    if text[0].isdigit():
        return False
    if text.startswith(("•", "-", "*")):
        return False
    if len(text.split()) < 3:
        return False
    return True

# ============================================================
# Pydantic Schemas
# ============================================================

class AgenticChunk(BaseModel):
    chunk_title: str
    chunk_text: str
    chunk_summary: str
    chunk_type: str = Field(description="concept | how_to | reference")
    primary_topic: str
    keywords: List[str]
    difficulty: str = Field(description="beginner | intermediate | advanced")

class AgenticChunkList(BaseModel):
    chunks: List[AgenticChunk]

# ============================================================
# PROMPT
# ============================================================

AGENTIC_CHUNK_PROMPT = f"""
You are preparing chunks for a Retrieval-Augmented Generation (RAG) system
based on Microsoft Copilot Studio documentation.

CRITICAL RULES:
- You MUST select topics ONLY from the list below
- DO NOT invent, rename, or generalize topics
- Select exactly ONE primary_topic
- If a concept does not clearly match an allowed topic,choose the closest applicable one.NEVER invent new topic names.

Allowed topics:
{COPILOT_TOPICS}

Return JSON under key "chunks".

Text:
<<<
{{TEXT}}
>>>
"""

# ============================================================
# VALIDATION
# ============================================================

def validate_topic(topic: str):
    if topic not in COPILOT_TOPICS:
        raise ValueError(f"Invalid primary_topic: {topic}")

# ============================================================
# PDF → H2 SECTION EXTRACTION
# ============================================================

def extract_h2_sections(pdf_path: str):
    doc = fitz.open(pdf_path)
    sections = []
    current = None

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" not in block:
                continue

            for line in block["lines"]:
                text = " ".join(span["text"] for span in line["spans"]).strip()
                if not text:
                    continue

                font_size = max(span["size"] for span in line["spans"])

                if is_h2(text, font_size):
                    if current:
                        sections.append(current)
                    current = {
                        "title": text,
                        "text": "",
                        "page_start": page_num,
                        "page_end": page_num,
                    }
                elif current:
                    current["text"] += text + "\n"
                    current["page_end"] = page_num

    if current:
        sections.append(current)

    return sections

# ============================================================
# BATCHING
# ============================================================

def batch_sections(sections, batch_size):
    batches, current = [], []
    for s in sections:
        current.append(s)
        if len(current) == batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches

# ============================================================
# CACHE
# ============================================================

def cache_key(text: str):
    return hashlib.sha256(text.encode()).hexdigest()

def load_cache(text: str):
    path = CACHE_DIR / cache_key(text)
    if path.exists():
        return json.loads(path.read_text())
    return None

def save_cache(text: str, data):
    path = CACHE_DIR / cache_key(text)
    path.write_text(json.dumps(data, indent=2))

# ============================================================
# LLM PROCESSING
# ============================================================

def process_batch(batch, llm):
    combined_text = "\n\n".join(
        f"{s['title']}\n{s['text']}" for s in batch
    )

    cached = load_cache(combined_text)
    if cached:
        return cached

    result = llm.invoke([
        HumanMessage(content=AGENTIC_CHUNK_PROMPT.format(TEXT=combined_text))
    ])

    data = result.model_dump()
    save_cache(combined_text, data)
    return data

# ============================================================
# MAIN INGESTION PIPELINE
# ============================================================

def ingest_pdf_fast():
    print("📄 Extracting H2 sections...")
    sections = extract_h2_sections(PDF_PATH)
    print(f"✔ Found {len(sections)} H2 sections")

    batches = batch_sections(sections, H2_BATCH_SIZE)
    print(f"✔ Created {len(batches)} LLM batches")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    ).with_structured_output(AgenticChunkList)

    all_docs: List[Document] = []

    print("🧠 Processing batches (parallel)...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_batch, b, llm) for b in batches]

        for f in as_completed(futures):
            data = f.result()
            for c in data["chunks"]:
                try:
                    chunk = AgenticChunk(**c)

                    # 🔒 STRICT but safe: drop invalid-topic chunks
                    validate_topic(chunk.primary_topic)

                except ValueError as e:
                # Optional: uncomment to see what was dropped
                # print(f"Dropping chunk due to topic error: {e}")
                    continue

                all_docs.append(
                    Document(
                        page_content=chunk.chunk_text,
                        metadata={
                            "title": chunk.chunk_title,
                            "summary": chunk.chunk_summary,
                            "primary_topic": chunk.primary_topic,
                            "chunk_type": chunk.chunk_type,
                            "keywords": chunk.keywords,
                            "difficulty": chunk.difficulty,
                            "source": PDF_PATH
                        }
                    )
                )


    print(f"📦 Total chunks created: {len(all_docs)}")

    # ========================================================
    # MILVUS INGESTION
    # ========================================================

    print("🔢 Embedding and storing in Milvus...")



    vectorstore = Milvus(
        embedding_function=OpenAIEmbeddings(),
        collection_name="rag_chunks",
        connection_args={
            "host": "localhost",
            "port": "19530"
        }
    )



    texts = []
    metadatas = []

    for i, doc in enumerate(all_docs):
        texts.append(doc.page_content)
        metadatas.append({
            "source": doc.metadata["source"],
            "chunk_id": i,
            "primary_topic": doc.metadata["primary_topic"]
        })

    vectorstore.add_texts(
        texts=texts,
        metadatas=metadatas
    )

    print("✅ FAST INGESTION COMPLETE (Milvus)")
    print(f"📚 Collection: {COLLECTION_NAME}")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    ingest_pdf_fast()
