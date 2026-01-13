# FAST Copilot Studio PDF Ingestion (H2-based + Batched + Cached)

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
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# ---------- Pydantic ----------
from pydantic import BaseModel, Field

# ---------- Config ----------
from config import POSTGRES_URL


# ============================================================
# CONFIG
# ============================================================

PDF_PATH = r"D:\RAG_Agent_Internship\data\copilot_studio.pdf"
NEW_COLLECTION = "copilot_chunks_v2_fast_h2"

MAX_WORKERS = 4          # parallel LLM calls
H2_BATCH_SIZE = 3        # H2 sections per LLM call

CACHE_DIR = Path(".llm_cache")
CACHE_DIR.mkdir(exist_ok=True)


# ============================================================
# CONTROLLED TOPIC TAXONOMY (NO LLM CREATION)
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

def is_h2(text: str, font_size: float) -> bool:
    # Font size threshold (still required)
    if font_size < 15:
        return False

    # Too long → probably paragraph text
    if len(text) > 120:
        return False

    # Headings usually don't end with punctuation
    if text.endswith((".", ":", ";", ",")):
        return False

    # Exclude numbered steps like "1. Do this"
    if text[0].isdigit():
        return False

    # Exclude bullet points
    if text.startswith(("•", "-", "*")):
        return False

    # Exclude very short noise (e.g. UI labels)
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
    secondary_topics: List[str]

    keywords: List[str]
    difficulty: str = Field(description="beginner | intermediate | advanced")


class AgenticChunkList(BaseModel):
    chunks: List[AgenticChunk]


# ============================================================
# PROMPT (LLM ONLY MAPS TO ALLOWED TOPICS)
# ============================================================

AGENTIC_CHUNK_PROMPT = f"""
You are preparing chunks for a Retrieval-Augmented Generation (RAG) system
based on Microsoft Copilot Studio documentation.

CRITICAL RULES:
- You MUST select topics ONLY from the list below
- DO NOT invent, rename, or generalize topics
- Topics must reflect how users would ask questions

Allowed topics:
{COPILOT_TOPICS}

Instructions:
- Split text into semantically meaningful chunks
- One Copilot Studio concept per chunk
- Do NOT split steps or explanations mid-way
- Select exactly ONE primary_topic
- Select 0–3 secondary_topics

Return JSON under key "chunks".

Text:
<<<
{{TEXT}}
>>>
"""


# ============================================================
# VALIDATION
# ============================================================
TOPIC_ALIASES = {
    "entities": "data_sources",
    "entities": "data_sources",
    "speech synthesis": "generative_ai_flows",

}

def validate_topics(chunk: AgenticChunk):
    # ---- PRIMARY TOPIC (STRICT) ----
    if chunk.primary_topic in TOPIC_ALIASES:
        chunk.primary_topic = TOPIC_ALIASES[chunk.primary_topic]

    if chunk.primary_topic not in COPILOT_TOPICS:
        raise ValueError(f"Invalid primary_topic: {chunk.primary_topic}")

    # ---- SECONDARY TOPICS (LENIENT) ----
    cleaned_secondary = []

    for t in chunk.secondary_topics:
        # normalize aliases
        if t in TOPIC_ALIASES:
            t = TOPIC_ALIASES[t]

        # keep only known topics, DROP the rest
        if t in COPILOT_TOPICS:
            cleaned_secondary.append(t)
        else:
            # optional: log once if you want
            # print("Dropping secondary topic:", t)
            pass

    chunk.secondary_topics = cleaned_secondary




# ============================================================
# PDF → H2 SECTION EXTRACTION
# ============================================================

def extract_h2_sections(pdf_path: str):
    import fitz  # PyMuPDF

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
                chunk = AgenticChunk(**c)
                validate_topics(chunk)

                all_docs.append(
                    Document(
                        page_content=chunk.chunk_text,
                        metadata={
                            "title": chunk.chunk_title,
                            "summary": chunk.chunk_summary,
                            "primary_topic": chunk.primary_topic,
                            "secondary_topics": chunk.secondary_topics,
                            "chunk_type": chunk.chunk_type,
                            "keywords": chunk.keywords,
                            "difficulty": chunk.difficulty,
                            "source": PDF_PATH
                        }
                    )
                )

    print(f"📦 Total chunks created: {len(all_docs)}")

    print("🔢 Embedding and storing in PGVector...")
    db = PGVector(
        connection_string=POSTGRES_URL,
        embedding_function=OpenAIEmbeddings(),
        collection_name=NEW_COLLECTION
    )

    db.add_documents(all_docs)

    print("✅ FAST INGESTION COMPLETE")
    print(f"📚 Collection: {NEW_COLLECTION}")
    print("🛡️ Old embeddings untouched")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    ingest_pdf_fast()
