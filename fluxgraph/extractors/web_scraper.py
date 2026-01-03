"""
FluxGraph Web Scraper

Production-grade web scraping with just a URL:
- Automatic content extraction
- JavaScript rendering support
- Smart content cleaning
- Metadata extraction
- Rate limiting and retries
- Proxy support
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin, urlparse
from datetime import datetime
import hashlib

try:
    import httpx
except ImportError:
    httpx = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from markdownify import markdownify as md
except ImportError:
    md = None


logger = logging.getLogger(__name__)


class ScrapedContent:
    """Container for scraped content"""

    def __init__(
        self,
        url: str,
        title: Optional[str] = None,
        text: Optional[str] = None,
        markdown: Optional[str] = None,
        html: Optional[str] = None,
        metadata: Optional[Dict] = None,
        links: Optional[List[str]] = None,
        images: Optional[List[str]] = None,
        timestamp: Optional[datetime] = None
    ):
        self.url = url
        self.title = title
        self.text = text
        self.markdown = markdown
        self.html = html
        self.metadata = metadata or {}
        self.links = links or []
        self.images = images or []
        self.timestamp = timestamp or datetime.utcnow()

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "markdown": self.markdown,
            "metadata": self.metadata,
            "links": self.links,
            "images": self.images,
            "timestamp": self.timestamp.isoformat(),
            "content_hash": self.content_hash()
        }

    def content_hash(self) -> str:
        """Generate hash of content for deduplication"""
        content = (self.text or "") + (self.title or "")
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class WebScraper:
    """
    Production-grade web scraper

    Example:
        scraper = WebScraper()

        # Simple scraping
        content = await scraper.scrape("https://example.com")
        print(content.title)
        print(content.text)

        # Batch scraping
        results = await scraper.scrape_batch([
            "https://example.com",
            "https://another.com"
        ])

        # With options
        content = await scraper.scrape(
            "https://example.com",
            extract_links=True,
            extract_images=True,
            clean_text=True
        )
    """

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        user_agent: Optional[str] = None,
        headers: Optional[Dict] = None,
        proxy: Optional[str] = None,
        follow_redirects: bool = True
    ):
        """
        Initialize web scraper

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            user_agent: Custom user agent
            headers: Additional HTTP headers
            proxy: Proxy URL
            follow_redirects: Follow HTTP redirects
        """
        if httpx is None:
            raise ImportError("httpx is required for web scraping. Install with: pip install httpx")

        if BeautifulSoup is None:
            raise ImportError("beautifulsoup4 is required. Install with: pip install beautifulsoup4")

        self.timeout = timeout
        self.max_retries = max_retries
        self.follow_redirects = follow_redirects

        self.headers = headers or {}
        self.headers.setdefault(
            "User-Agent",
            user_agent or "FluxGraph/2.3.0 (Web Scraper; +https://github.com/ihtesham-jahangir/fluxgraph)"
        )

        self.client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=follow_redirects,
            headers=self.headers,
            proxies=proxy
        )

    async def scrape(
        self,
        url: str,
        extract_links: bool = False,
        extract_images: bool = False,
        extract_metadata: bool = True,
        clean_text: bool = True,
        convert_to_markdown: bool = False,
        max_content_length: Optional[int] = None
    ) -> ScrapedContent:
        """
        Scrape a single URL

        Args:
            url: URL to scrape
            extract_links: Extract all links from page
            extract_images: Extract all image URLs
            extract_metadata: Extract meta tags and page metadata
            clean_text: Clean and format extracted text
            convert_to_markdown: Convert HTML to markdown
            max_content_length: Maximum content length (truncate if exceeded)

        Returns:
            ScrapedContent object

        Raises:
            httpx.HTTPError: If request fails
            Exception: For other errors
        """
        for attempt in range(self.max_retries):
            try:
                response = await self.client.get(url)
                response.raise_for_status()

                html = response.text
                soup = BeautifulSoup(html, 'html.parser')

                # Extract title
                title = self._extract_title(soup)

                # Extract text
                text = self._extract_text(soup, clean=clean_text)

                if max_content_length and len(text) > max_content_length:
                    text = text[:max_content_length] + "..."

                # Convert to markdown
                markdown_content = None
                if convert_to_markdown:
                    markdown_content = self._html_to_markdown(html)

                # Extract metadata
                metadata = {}
                if extract_metadata:
                    metadata = self._extract_metadata(soup, response)

                # Extract links
                links = []
                if extract_links:
                    links = self._extract_links(soup, url)

                # Extract images
                images = []
                if extract_images:
                    images = self._extract_images(soup, url)

                logger.info(f"✅ Scraped: {url}")

                return ScrapedContent(
                    url=url,
                    title=title,
                    text=text,
                    markdown=markdown_content,
                    html=html,
                    metadata=metadata,
                    links=links,
                    images=images
                )

            except httpx.HTTPError as e:
                logger.warning(f"⚠️ Attempt {attempt + 1}/{self.max_retries} failed for {url}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

            except Exception as e:
                logger.error(f"❌ Error scraping {url}: {e}")
                raise

    async def scrape_batch(
        self,
        urls: List[str],
        max_concurrent: int = 5,
        **scrape_kwargs
    ) -> List[ScrapedContent]:
        """
        Scrape multiple URLs concurrently

        Args:
            urls: List of URLs to scrape
            max_concurrent: Maximum concurrent requests
            **scrape_kwargs: Arguments passed to scrape()

        Returns:
            List of ScrapedContent objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def scrape_with_semaphore(url: str):
            async with semaphore:
                try:
                    return await self.scrape(url, **scrape_kwargs)
                except Exception as e:
                    logger.error(f"❌ Failed to scrape {url}: {e}")
                    return None

        results = await asyncio.gather(
            *[scrape_with_semaphore(url) for url in urls],
            return_exceptions=False
        )

        # Filter out None results
        return [r for r in results if r is not None]

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page title"""
        # Try <title> tag
        if soup.title and soup.title.string:
            return soup.title.string.strip()

        # Try Open Graph title
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title["content"].strip()

        # Try <h1> tag
        h1 = soup.find("h1")
        if h1:
            return h1.get_text().strip()

        return None

    def _extract_text(self, soup: BeautifulSoup, clean: bool = True) -> str:
        """Extract main text content"""
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Try to find main content area
        main_content = (
            soup.find("main") or
            soup.find("article") or
            soup.find("div", class_=re.compile(r"(content|main|article|post)", re.I)) or
            soup.body
        )

        if main_content:
            text = main_content.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        if clean:
            # Clean up whitespace
            lines = [line.strip() for line in text.split("\n")]
            lines = [line for line in lines if line]  # Remove empty lines
            text = "\n".join(lines)

        return text

    def _html_to_markdown(self, html: str) -> Optional[str]:
        """Convert HTML to markdown"""
        if md is None:
            logger.warning("markdownify not installed, skipping markdown conversion")
            return None

        try:
            return md(html, heading_style="ATX", bullets="-")
        except Exception as e:
            logger.error(f"Markdown conversion failed: {e}")
            return None

    def _extract_metadata(self, soup: BeautifulSoup, response: Any) -> Dict:
        """Extract page metadata"""
        metadata = {
            "content_type": response.headers.get("content-type", ""),
            "status_code": response.status_code,
            "final_url": str(response.url)
        }

        # Meta description
        description = soup.find("meta", attrs={"name": "description"})
        if description and description.get("content"):
            metadata["description"] = description["content"].strip()

        # Meta keywords
        keywords = soup.find("meta", attrs={"name": "keywords"})
        if keywords and keywords.get("content"):
            metadata["keywords"] = keywords["content"].strip()

        # Open Graph data
        og_tags = soup.find_all("meta", property=re.compile(r"^og:"))
        for tag in og_tags:
            key = tag.get("property", "").replace("og:", "og_")
            value = tag.get("content", "")
            if key and value:
                metadata[key] = value

        # Author
        author = soup.find("meta", attrs={"name": "author"})
        if author and author.get("content"):
            metadata["author"] = author["content"].strip()

        # Published date
        published = soup.find("meta", property="article:published_time")
        if published and published.get("content"):
            metadata["published_time"] = published["content"]

        return metadata

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all links from page"""
        links = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            links.append(absolute_url)

        # Deduplicate
        return list(set(links))

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all image URLs"""
        images = []

        for img in soup.find_all("img", src=True):
            src = img["src"]
            absolute_url = urljoin(base_url, src)
            images.append(absolute_url)

        # Deduplicate
        return list(set(images))

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class SmartScraper(WebScraper):
    """
    Smart scraper with automatic content detection

    Automatically detects:
    - Article content
    - Product pages
    - Documentation pages
    - Forums/discussions
    """

    async def scrape_article(self, url: str) -> Dict[str, Any]:
        """
        Scrape article with structured extraction

        Returns:
            Dictionary with: title, author, published_date, content, images, tags
        """
        content = await self.scrape(
            url,
            extract_metadata=True,
            extract_images=True,
            clean_text=True
        )

        return {
            "title": content.title,
            "author": content.metadata.get("author"),
            "published_date": content.metadata.get("published_time"),
            "description": content.metadata.get("description"),
            "content": content.text,
            "images": content.images,
            "tags": content.metadata.get("keywords", "").split(",") if content.metadata.get("keywords") else [],
            "url": url
        }

    async def scrape_sitemap(self, sitemap_url: str) -> List[str]:
        """
        Extract URLs from sitemap

        Args:
            sitemap_url: URL to sitemap.xml

        Returns:
            List of URLs found in sitemap
        """
        response = await self.client.get(sitemap_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "xml") or BeautifulSoup(response.text, "html.parser")

        urls = []
        for loc in soup.find_all("loc"):
            urls.append(loc.get_text().strip())

        logger.info(f"📋 Found {len(urls)} URLs in sitemap")
        return urls


# Convenience functions

async def scrape_url(
    url: str,
    **kwargs
) -> ScrapedContent:
    """
    Quick scraping function

    Example:
        from fluxgraph.extractors.web_scraper import scrape_url

        content = await scrape_url("https://example.com")
        print(content.title)
        print(content.text)
    """
    async with WebScraper() as scraper:
        return await scraper.scrape(url, **kwargs)


async def scrape_urls(
    urls: List[str],
    max_concurrent: int = 5,
    **kwargs
) -> List[ScrapedContent]:
    """
    Quick batch scraping

    Example:
        urls = ["https://example.com", "https://another.com"]
        results = await scrape_urls(urls)
    """
    async with WebScraper() as scraper:
        return await scraper.scrape_batch(urls, max_concurrent=max_concurrent, **kwargs)


async def extract_article(url: str) -> Dict[str, Any]:
    """
    Quick article extraction

    Example:
        article = await extract_article("https://blog.example.com/post")
        print(article["title"])
        print(article["author"])
        print(article["content"])
    """
    async with SmartScraper() as scraper:
        return await scraper.scrape_article(url)
