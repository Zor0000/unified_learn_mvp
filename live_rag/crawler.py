import asyncio
from crawl4ai import AsyncWebCrawler

def fetch_with_crawl4ai(url: str) -> str:
    async def _crawl():
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url)
            return result.markdown

    return asyncio.run(_crawl())
