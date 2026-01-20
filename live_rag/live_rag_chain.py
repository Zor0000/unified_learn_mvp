import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from live_rag.crawler import fetch_with_crawl4ai

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def live_rag_crawl4ai(query: str, url: str):
    total_start = time.time()

    # Crawl latency
    crawl_start = time.time()
    web_content = fetch_with_crawl4ai(url)
    crawl_latency = time.time() - crawl_start

    # LLM latency
    llm_start = time.time()
    messages = [
        SystemMessage(
            content=(
        "Answer the question using the provided website content. "
        "You may summarize, rephrase, or infer information that is clearly implied "
        "by the content. If the information is not available at all, say "
        "'The information is not explicitly mentioned on this page.'"
            )
        ),
        HumanMessage(
            content=f"""
Website Content:
{web_content[:12000]}

Question:
{query}
"""
        )
    ]
    response = llm.invoke(messages)
    llm_latency = time.time() - llm_start

    total_latency = time.time() - total_start

    return {
        "answer": response.content,
        "crawl_latency": crawl_latency,
        "llm_latency": llm_latency,
        "total_latency": total_latency
    }
