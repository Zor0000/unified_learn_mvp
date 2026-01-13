import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.prompts import ChatPromptTemplate
from config import POSTGRES_URL

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

Explain things clearly and naturally, like ChatGPT would.

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
# LIGHTWEIGHT TOPIC ROUTING (VERY IMPORTANT)
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

    return None  # fallback = no topic filter


# ======================================================
# CONFIG
# ======================================================

COLLECTION_NAME = "copilot_chunks_v2_fast_h2"


def start_chat():
    embeddings = OpenAIEmbeddings()

    vectorstore = PGVector(
        connection_string=POSTGRES_URL,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4
    )

    print("\n🧠 Copilot Studio RAG Chat Ready (type 'exit' to quit)\n")

    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            break

        question_type = detect_question_type(question)
        topic = detect_topic(question)

        # -------------------------------
        # Choose retriever
        # -------------------------------
        search_kwargs = {"k": 6}

        if topic:
            search_kwargs["filter"] = {
                "primary_topic": topic
            }

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
        # Retrieve documents
        # -------------------------------
        docs = retriever.invoke(question)

        if not docs:
            print("\n🤖 Answer:\n")
            print("I don’t have enough information in the knowledge base to answer this.")
            print("\n" + "-" * 50 + "\n")
            continue

        # Order chunks logically if available
        docs = sorted(docs, key=lambda d: d.metadata.get("page_number", 0))

        # -------------------------------
        # Build structured context
        # -------------------------------
        context_parts = []
        for i, d in enumerate(docs, start=1):
            context_parts.append(
                f"[Source {i}]\n{d.page_content}"
            )

        context = "\n\n".join(context_parts)

        # -------------------------------
        # Prompt → LLM
        # -------------------------------
        messages = prompt.format_messages(
            context=context,
            question=question
        )

        response = llm.invoke(messages)

        print("\n🤖 Answer:\n")
        print(response.content.strip())
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    start_chat()
