from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_milvus import Milvus
from langsmith import traceable

from rag.chat import (
    detect_question_type,
    detect_topic,
    HOW_PROMPT,
    WHAT_PROMPT,
    WHY_PROMPT,
    DEFAULT_PROMPT,
)

# ==============================
# MILVUS CONFIG
# ==============================

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "rag_chunks"


@traceable(name="rag_answer_milvus")
def run_rag_question(question: str) -> dict:
    embeddings = OpenAIEmbeddings()

    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"uri": MILVUS_URI}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4
    )

    # Intent detection (kept for prompts)
    question_type = detect_question_type(question)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}
    )

    if question_type == "how":
        prompt = HOW_PROMPT
    elif question_type == "what":
        prompt = WHAT_PROMPT
    elif question_type == "why":
        prompt = WHY_PROMPT
    else:
        prompt = DEFAULT_PROMPT

    docs = retriever.invoke(question)
    contexts = [d.page_content for d in docs]

    if not contexts:
        return {
            "question": question,
            "contexts": [],
            "answer": "I don’t have enough information in the provided context."
        }

    messages = prompt.format_messages(
        context="\n\n".join(contexts),
        question=question
    )

    response = llm.invoke(messages)

    return {
        "question": question,
        "contexts": contexts,
        "answer": response.content.strip()
    }

