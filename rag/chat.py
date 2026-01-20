import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_core.prompts import ChatPromptTemplate

# ======================================================
# PROMPTS
# ======================================================

HOW_PROMPT = ChatPromptTemplate.from_template(
    """
You are an expert Copilot Studio assistant.

Using ONLY the context below, answer the question in a clear,
user-friendly way.

Guidelines:
- Start with a short explanation of what the process is
- Then provide step-by-step instructions
- Use clear numbering
- Explain *why* a step is needed if helpful
- Do NOT invent steps not present in the context

Context:
{context}

Question:
{question}
"""
)

WHAT_PROMPT = ChatPromptTemplate.from_template(
    """
You are an expert Copilot Studio assistant.

Using ONLY the context below:
- Clearly define the concept
- Explain its purpose
- Mention key components or ideas
- Keep it easy to understand

Context:
{context}

Question:
{question}
"""
)

WHY_PROMPT = ChatPromptTemplate.from_template(
    """
You are an expert Copilot Studio assistant.

Using ONLY the context below:
- Explain the reasoning, benefits, or motivation
- Structure the explanation logically
- Keep it concise but insightful

Context:
{context}

Question:
{question}
"""
)

DEFAULT_PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful Copilot Studio assistant.

Answer the question using ONLY the context below.
If the answer is not clearly present, say:
"I don’t have enough information in the provided context."

Explain things clearly and naturally.

Context:
{context}

Question:
{question}
"""
)

# ======================================================
# SIMPLE INTENT DETECTION
# ======================================================

def detect_question_type(question: str) -> str:
    q = question.lower().strip()

    if q.startswith(("how", "steps", "procedure", "process")):
        return "how"
    elif q.startswith(("what", "define", "meaning")):
        return "what"
    elif q.startswith(("why", "benefit", "reason", "advantage")):
        return "why"
    else:
        return "default"

# ======================================================
# LIGHTWEIGHT TOPIC ROUTING
# ======================================================

def detect_topic(question: str) -> str | None:
    q = question.lower()

    if "orchestration" in q:
        return "agent_orchestration"
    if "deploy" in q or "publish" in q:
        return "deployment_and_publishing"
    if "connect" in q or "api" in q:
        return "external_apis"
    if "security" in q or "auth" in q:
        return "authentication_and_security"
    if "monitor" in q or "analytics" in q:
        return "monitoring_and_analytics"

    return None

# ======================================================
# CONFIG
# ======================================================

COLLECTION_NAME = "rag_chunks"

def start_chat():
    embeddings = OpenAIEmbeddings()

    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={
            "host": "localhost",
            "port": "19530"
        }
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4
    )

    print("\n🧠 Copilot Studio RAG Chat (Milvus) — type 'exit' to quit\n")

    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            break

        total_start = time.time()  # ⏱️ TOTAL TIMER

        question_type = detect_question_type(question)
        topic = detect_topic(question)

        search_kwargs = {"k": 6}

        if topic:
            search_kwargs["expr"] = f'primary_topic == "{topic}"'

        retriever = vectorstore.as_retriever(
            search_kwargs=search_kwargs
        )

        if question_type == "how":
            prompt = HOW_PROMPT
        elif question_type == "what":
            prompt = WHAT_PROMPT
        elif question_type == "why":
            prompt = WHY_PROMPT
        else:
            prompt = DEFAULT_PROMPT

        # -------------------------------
        # ⏱️ RETRIEVAL LATENCY
        # -------------------------------
        retrieval_start = time.time()
        docs = retriever.invoke(question)
        retrieval_latency = time.time() - retrieval_start

        if not docs:
            print("\n🤖 Answer:\n")
            print("I don’t have enough information in the knowledge base to answer this.")
            print(f"\n⏱️ Retrieval latency: {retrieval_latency:.2f}s")
            print("\n" + "-" * 50 + "\n")
            continue

        context_parts = []
        for i, d in enumerate(docs, start=1):
            context_parts.append(f"[Source {i}]\n{d.page_content}")

        context = "\n\n".join(context_parts)

        messages = prompt.format_messages(
            context=context,
            question=question
        )

        # -------------------------------
        # ⏱️ LLM LATENCY
        # -------------------------------
        llm_start = time.time()
        response = llm.invoke(messages)
        llm_latency = time.time() - llm_start

        total_latency = time.time() - total_start

        print("\n🤖 Answer:\n")
        print(response.content.strip())

        print(
            f"\n⏱️ Retrieval latency: {retrieval_latency:.2f}s"
            f"\n⏱️ LLM latency: {llm_latency:.2f}s"
            f"\n⏱️ Total latency: {total_latency:.2f}s"
        )

        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    start_chat()
