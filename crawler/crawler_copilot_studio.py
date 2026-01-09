import json
from datetime import datetime
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
import asyncio


OUTPUT_FILE = "data/processed/copilot_studio_docs.json"
START_URL = "https://learn.microsoft.com/en-us/microsoft-copilot-studio"


def extract_structured_content(html: str, url: str):
    """
    Extract headings + paragraphs from Microsoft Docs HTML
    and return structured JSON.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Main content area for Microsoft Docs
    main = soup.find("main")
    if not main:
        return None

    title = soup.title.text.strip() if soup.title else ""

    sections = []
    current_section = None

    for tag in main.find_all(["h1", "h2", "h3", "p", "li", "pre"]):
        if tag.name in ["h1", "h2", "h3"]:
            if current_section:
                sections.append(current_section)

            current_section = {
                "heading": tag.get_text(strip=True),
                "content": ""
            }

        else:
            text = tag.get_text(" ", strip=True)
            if not text:
                continue

            if not current_section:
                current_section = {
                    "heading": "Introduction",
                    "content": ""
                }

            current_section["content"] += text + "\n"

    if current_section:
        sections.append(current_section)

    return {
        "source": url,
        "title": title,
        "sections": sections,
        "metadata": {
            "product": "microsoft-copilot-studio",
            "doc_type": "documentation",
            "language": "en",
            "crawled_at": datetime.utcnow().isoformat()
        }
    }


async def crawl():
    crawler = AsyncWebCrawler(
        max_depth=2,          # crawl child pages
        same_domain=True,
        respect_robots_txt=True
    )

    results = []

    async with crawler:
        pages = await crawler.arun(START_URL)

        for page in pages:
            if not page.success or not page.html:
                continue

            structured = extract_structured_content(page.html, page.url)
            if structured:
                results.append(structured)

    # Save to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Crawled {len(results)} pages")
    print(f"📁 Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(crawl())
