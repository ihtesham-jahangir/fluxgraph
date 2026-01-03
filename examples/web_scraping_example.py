"""
FluxGraph Web Scraping & Document Extraction Examples

Demonstrates:
- URL scraping
- Batch scraping
- Article extraction
- Document extraction
- Data parsing
"""

import asyncio
from fluxgraph.extractors.web_scraper import (
    WebScraper,
    SmartScraper,
    scrape_url,
    scrape_urls,
    extract_article
)
from fluxgraph.extractors.document_extractor import (
    DocumentExtractor,
    extract_text,
    extract_from_url
)
from fluxgraph.extractors.data_parser import (
    DataParser,
    extract_all_data
)


# ============================================================================
# Example 1: Simple URL Scraping
# ============================================================================

async def example_simple_scraping():
    """Scrape a single URL"""
    print("="*80)
    print("Example 1: Simple URL Scraping")
    print("="*80)

    # Quick scraping
    content = await scrape_url("https://example.com")

    print(f"Title: {content.title}")
    print(f"Text Preview: {content.text[:200]}...")
    print(f"Word Count: {len(content.text.split())}")


# ============================================================================
# Example 2: Batch Scraping
# ============================================================================

async def example_batch_scraping():
    """Scrape multiple URLs at once"""
    print("\n" + "="*80)
    print("Example 2: Batch Scraping (Multiple URLs)")
    print("="*80)

    urls = [
        "https://example.com",
        "https://httpbin.org/html",
        "https://www.iana.org"
    ]

    results = await scrape_urls(urls, max_concurrent=3)

    for content in results:
        print(f"\n✅ {content.url}")
        print(f"   Title: {content.title}")
        print(f"   Length: {len(content.text)} chars")


# ============================================================================
# Example 3: Advanced Scraping with Options
# ============================================================================

async def example_advanced_scraping():
    """Scraping with full options"""
    print("\n" + "="*80)
    print("Example 3: Advanced Scraping")
    print("="*80)

    async with WebScraper() as scraper:
        content = await scraper.scrape(
            "https://example.com",
            extract_links=True,
            extract_images=True,
            extract_metadata=True,
            convert_to_markdown=True
        )

        print(f"Title: {content.title}")
        print(f"Description: {content.metadata.get('description', 'N/A')}")
        print(f"Links found: {len(content.links)}")
        print(f"Images found: {len(content.images)}")

        if content.markdown:
            print(f"\nMarkdown Preview:\n{content.markdown[:300]}...")


# ============================================================================
# Example 4: Article Extraction
# ============================================================================

async def example_article_extraction():
    """Extract article with structured data"""
    print("\n" + "="*80)
    print("Example 4: Smart Article Extraction")
    print("="*80)

    # This would work on actual blog/news sites
    article = await extract_article("https://example.com")

    print(f"Title: {article['title']}")
    print(f"Author: {article['author']}")
    print(f"Published: {article['published_date']}")
    print(f"Tags: {', '.join(article['tags'])}")
    print(f"\nContent Preview:\n{article['content'][:300]}...")


# ============================================================================
# Example 5: Document Extraction from URL
# ============================================================================

async def example_document_from_url():
    """Extract text from PDF/DOCX via URL"""
    print("\n" + "="*80)
    print("Example 5: Document Extraction from URL")
    print("="*80)

    # Example: Extract from a PDF URL
    # pdf_url = "https://example.com/document.pdf"
    # text = await extract_from_url(pdf_url)
    # print(f"Extracted {len(text)} characters from PDF")

    print("(This would extract text from PDF/DOCX URLs)")


# ============================================================================
# Example 6: Local Document Extraction
# ============================================================================

def example_local_document():
    """Extract text from local files"""
    print("\n" + "="*80)
    print("Example 6: Local Document Extraction")
    print("="*80)

    extractor = DocumentExtractor()

    # PDF
    # content = extractor.extract_from_file("document.pdf")
    # print(f"PDF: {content.metadata['page_count']} pages")

    # Word
    # content = extractor.extract_from_file("document.docx")
    # print(f"DOCX: {len(content.tables)} tables")

    # Excel
    # content = extractor.extract_from_file("spreadsheet.xlsx")
    # print(f"Excel: {content.metadata['sheet_count']} sheets")

    print("(This would extract from local PDF, DOCX, XLSX files)")


# ============================================================================
# Example 7: Data Parsing
# ============================================================================

def example_data_parsing():
    """Extract structured data from text"""
    print("\n" + "="*80)
    print("Example 7: Data Parsing & Extraction")
    print("="*80)

    text = """
    Contact us at support@example.com or sales@company.com
    Visit https://example.com or https://company.io
    Call us at 555-123-4567 or +1-800-555-0199
    Prices: $99.99, $1,234.56, $5,000
    Published on 2026-01-02
    Tags: #ai #machinelearning #python
    Mention @fluxgraph for questions
    """

    # Extract all data types
    data = extract_all_data(text)

    print(f"Emails: {data['emails']}")
    print(f"URLs: {data['urls']}")
    print(f"Phone Numbers: {data['phone_numbers']}")
    print(f"Prices: {data['prices']}")
    print(f"Dates: {data['dates']}")
    print(f"Hashtags: {data['hashtags']}")
    print(f"Mentions: {data['mentions']}")


# ============================================================================
# Example 8: JSON/CSV Parsing
# ============================================================================

def example_structured_parsing():
    """Parse JSON, CSV, YAML"""
    print("\n" + "="*80)
    print("Example 8: Structured Data Parsing")
    print("="*80)

    parser = DataParser()

    # JSON
    json_text = '{"name": "FluxGraph", "version": "2.3.0", "features": ["scraping", "extraction"]}'
    data = parser.parse_json(json_text)
    print(f"JSON: {data}")

    # CSV
    csv_text = """name,age,city
John,30,NYC
Jane,25,LA"""
    data = parser.parse_csv(csv_text)
    print(f"\nCSV: {data}")

    # Extract key-value pairs
    kv_text = """
    name: FluxGraph
    version: 2.3.0
    author: Ihtesham Jahangir
    """
    data = parser.extract_key_values(kv_text)
    print(f"\nKey-Values: {data}")


# ============================================================================
# Example 9: Integration with FluxGraph Agents
# ============================================================================

async def example_agent_integration():
    """Use scraping in FluxGraph agents"""
    print("\n" + "="*80)
    print("Example 9: Integration with FluxGraph Agents")
    print("="*80)

    from fluxgraph.core.app import FluxApp

    app = FluxApp()

    @app.agent()
    async def web_researcher(url: str, **kwargs) -> dict:
        """Agent that researches web content"""
        # Scrape the URL
        content = await scrape_url(url, clean_text=True)

        # Extract structured data
        data = extract_all_data(content.text)

        return {
            "title": content.title,
            "summary": content.text[:500],
            "links_found": len(content.links) if hasattr(content, 'links') else 0,
            "emails": data['emails'],
            "urls": data['urls']
        }

    # Test the agent
    result = await web_researcher.run(url="https://example.com")
    print(f"Research Result: {result}")


# ============================================================================
# Example 10: Sitemap Crawling
# ============================================================================

async def example_sitemap_crawling():
    """Crawl entire website from sitemap"""
    print("\n" + "="*80)
    print("Example 10: Sitemap Crawling")
    print("="*80)

    async with SmartScraper() as scraper:
        # Get all URLs from sitemap
        # urls = await scraper.scrape_sitemap("https://example.com/sitemap.xml")
        # print(f"Found {len(urls)} URLs in sitemap")

        # Scrape them all
        # results = await scraper.scrape_batch(urls[:10], max_concurrent=5)
        # print(f"Scraped {len(results)} pages")

        print("(This would crawl all pages from sitemap.xml)")


# ============================================================================
# Run All Examples
# ============================================================================

async def main():
    """Run all examples"""
    print("\n🚀 FluxGraph Web Scraping & Extraction Examples\n")

    # Async examples
    await example_simple_scraping()
    await example_batch_scraping()
    await example_advanced_scraping()
    await example_article_extraction()
    await example_document_from_url()
    await example_agent_integration()
    await example_sitemap_crawling()

    # Sync examples
    example_local_document()
    example_data_parsing()
    example_structured_parsing()

    print("\n" + "="*80)
    print("✅ All examples completed!")
    print("="*80)


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())
