import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.prompts import ChatPromptTemplate
from config import POSTGRES_URL
from langchain_core.prompts import ChatPromptTemplate

# ---------------- PROMPT TEMPLATES ----------------

HOW_PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful technical assistant.

Answer the question using ONLY the context below.
Do NOT add information that is not present in the context.

The question is asking for steps.

Instructions:
- Combine relevant instructions from the context
- Present the answer as a clear, ordered step-by-step list
- Avoid repeating the same step
- Limit the main steps to at most 5 items
- Add optional steps only if necessary

Context:
{context}

Question:
{question}
"""
)

WHAT_PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful technical assistant.

Answer the question using ONLY the context below.

Instructions:
- Provide a clear, concise definition
- Explain the purpose and key idea
- Do not provide steps unless explicitly asked

Context:
{context}

Question:
{question}
"""
)

WHY_PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful technical assistant.

Answer the question using ONLY the context below.

Instructions:
- Explain the reasoning or benefits clearly
- Focus on concepts, not procedures
- Keep the explanation structured but concise

Context:
{context}

Question:
{question}
"""
)

DEFAULT_PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.

Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""
)

# --------------------------------------------------
def detect_question_type(question: str) -> str:
    q = question.lower().strip()

    if q.startswith(("how", "steps", "procedure")):
        return "how"
    elif q.startswith(("what", "define")):
        return "what"
    elif q.startswith(("why", "benefit", "reason")):
        return "why"
    else:
        return "default"


COLLECTION_NAME = "copilot_pdf_agentic"


def start_chat():
    # Embeddings (must match ingestion)
    embeddings = OpenAIEmbeddings()

    # Vector store
    vectorstore = PGVector(
        connection_string=POSTGRES_URL,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5
    )

    print("\n🧠 RAG Chat Ready (type 'exit' to quit)\n")

    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break

        # -------------------------------
        # 1. Detect question type
        # -------------------------------
        question_type = detect_question_type(question)

        # -------------------------------
        # 2. Choose retriever strategy
        # -------------------------------
        if question_type == "how":
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 10}
            )
            prompt = HOW_PROMPT
        elif question_type == "what":
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 5}
            )
            prompt = WHAT_PROMPT
        elif question_type == "why":
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 5}
            )
            prompt = WHY_PROMPT
        else:
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 5}
            )
            prompt = DEFAULT_PROMPT

        # -------------------------------
        # 3. Retrieve documents
        # -------------------------------
        docs = retriever.invoke(question)

        # Ensure procedural order (important for steps)
        docs = sorted(docs, key=lambda d: d.metadata.get("page_number", 0))

        # -------------------------------
        # 4. DEBUG: Inspect retrieved chunks
        # -------------------------------
        print("\n--- Retrieved Chunks ---")
        for d in docs:
            print(
                f"Page {d.metadata.get('page_number')} | "
                f"Tag: {d.metadata.get('tag')} | "
                f"Type: {d.metadata.get('chunk_type')}"
            )
            print(d.page_content[:200])
            print("------------------------")

        # -------------------------------
        # 5. Build context
        # -------------------------------
        context = "\n\n".join(doc.page_content for doc in docs)

        # -------------------------------
        # 6. Format prompt
        # -------------------------------
        messages = prompt.format_messages(
            context=context,
            question=question
        )

        # -------------------------------
        # 7. Call LLM
        # -------------------------------
        response = llm.invoke(messages)

        print("\n🤖 Answer:\n")
        print(response.content)
        print("\n" + "-" * 50 + "\n")



if __name__ == "__main__":
    start_chat()
