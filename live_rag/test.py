from live_rag.live_rag_chain import live_rag_crawl4ai

result = live_rag_crawl4ai(
    query="What can you build using Microsoft Copilot Studio?",
    url="https://learn.microsoft.com/en-us/microsoft-copilot-studio/"
)

print(result["answer"])
print(f"Crawl latency: {result['crawl_latency']:.2f}s")
print(f"LLM latency: {result['llm_latency']:.2f}s")
print(f"Total latency: {result['total_latency']:.2f}s")
