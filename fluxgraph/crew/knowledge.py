# fluxgraph/crew/knowledge.py
"""
CrewAI-style knowledge sources: PDF, Web scraping, Documents
Agents can access external knowledge bases during execution.
"""
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class KnowledgeSourceType(Enum):
    """Types of knowledge sources."""
    PDF = "pdf"
    WEB = "web"
    MARKDOWN = "markdown"
    TEXT = "text"
    CSV = "csv"
    JSON = "json"
    API = "api"
    DATABASE = "database"


@dataclass
class KnowledgeSource:
    """
    Base class for knowledge sources.
    
    Example:
        pdf_knowledge = PDFKnowledge(file_path="research_paper.pdf")
        web_knowledge = WebKnowledge(urls=["https://example.com"])
    """
    source_type: KnowledgeSourceType
    description: Optional[str] = None
    _content: Optional[str] = None
    _metadata: Dict[str, Any] = field(default_factory=dict)
    
    async def load(self) -> str:
        """Load content from source."""
        raise NotImplementedError
    
    def get_content(self) -> str:
        """Get loaded content."""
        if self._content is None:
            raise RuntimeError("Content not loaded. Call load() first.")
        return self._content
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the source."""
        return self._metadata


class PDFKnowledge(KnowledgeSource):
    """
    PDF document knowledge source.
    
    Example:
        pdf = PDFKnowledge(file_path="report.pdf")
        await pdf.load()
        content = pdf.get_content()
    """
    
    def __init__(self, file_path: Union[str, Path], **kwargs):
        super().__init__(source_type=KnowledgeSourceType.PDF, **kwargs)
        self.file_path = Path(file_path)
    
    async def load(self) -> str:
        """Load PDF content."""
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("PyPDF2 required: pip install PyPDF2")
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.file_path}")
        
        logger.info(f"📄 Loading PDF: {self.file_path}")
        
        content = []
        with open(self.file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            self._metadata = {
                "pages": len(pdf_reader.pages),
                "file_name": self.file_path.name,
                "file_size": self.file_path.stat().st_size
            }
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                content.append(f"--- Page {page_num + 1} ---\n{text}")
        
        self._content = "\n\n".join(content)
        logger.info(f"✅ Loaded {len(pdf_reader.pages)} pages from PDF")
        
        return self._content


class WebKnowledge(KnowledgeSource):
    """
    Web scraping knowledge source.
    
    Example:
        web = WebKnowledge(
            urls=["https://example.com/article"],
            max_depth=2
        )
        await web.load()
    """
    
    def __init__(
        self,
        urls: List[str],
        max_depth: int = 1,
        follow_links: bool = False,
        **kwargs
    ):
        super().__init__(source_type=KnowledgeSourceType.WEB, **kwargs)
        self.urls = urls
        self.max_depth = max_depth
        self.follow_links = follow_links
    
    async def load(self) -> str:
        """Scrape web content."""
        try:
            import aiohttp
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("Required: pip install aiohttp beautifulsoup4")
        
        logger.info(f"🌐 Scraping {len(self.urls)} URLs")
        
        content = []
        
        async with aiohttp.ClientSession() as session:
            for url in self.urls:
                try:
                    async with session.get(url) as response:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        # Get text
                        text = soup.get_text()
                        
                        # Clean up
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        text = '\n'.join(chunk for chunk in chunks if chunk)
                        
                        content.append(f"--- URL: {url} ---\n{text}")
                        
                        logger.info(f"✅ Scraped: {url}")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to scrape {url}: {e}")
        
        self._content = "\n\n".join(content)
        self._metadata = {
            "urls_scraped": len(self.urls),
            "total_length": len(self._content)
        }
        
        return self._content


class MarkdownKnowledge(KnowledgeSource):
    """
    Markdown document knowledge source.
    
    Example:
        md = MarkdownKnowledge(file_path="docs/README.md")
        await md.load()
    """
    
    def __init__(self, file_path: Union[str, Path], **kwargs):
        super().__init__(source_type=KnowledgeSourceType.MARKDOWN, **kwargs)
        self.file_path = Path(file_path)
    
    async def load(self) -> str:
        """Load markdown content."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {self.file_path}")
        
        logger.info(f"📝 Loading Markdown: {self.file_path}")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self._content = f.read()
        
        self._metadata = {
            "file_name": self.file_path.name,
            "file_size": self.file_path.stat().st_size,
            "lines": len(self._content.splitlines())
        }
        
        logger.info(f"✅ Loaded {self._metadata['lines']} lines from Markdown")
        
        return self._content


class TextKnowledge(KnowledgeSource):
    """
    Plain text knowledge source.
    
    Example:
        txt = TextKnowledge(content="This is knowledge content...")
    """
    
    def __init__(self, content: Optional[str] = None, file_path: Optional[Union[str, Path]] = None, **kwargs):
        super().__init__(source_type=KnowledgeSourceType.TEXT, **kwargs)
        self._content = content
        self.file_path = Path(file_path) if file_path else None
    
    async def load(self) -> str:
        """Load text content."""
        if self._content:
            return self._content
        
        if self.file_path and self.file_path.exists():
            logger.info(f"📄 Loading text file: {self.file_path}")
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self._content = f.read()
            return self._content
        
        raise ValueError("Either content or file_path must be provided")


class CSVKnowledge(KnowledgeSource):
    """
    CSV data knowledge source.
    
    Example:
        csv = CSVKnowledge(file_path="data.csv")
        await csv.load()
    """
    
    def __init__(self, file_path: Union[str, Path], **kwargs):
        super().__init__(source_type=KnowledgeSourceType.CSV, **kwargs)
        self.file_path = Path(file_path)
    
    async def load(self) -> str:
        """Load CSV as formatted text."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required: pip install pandas")
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.file_path}")
        
        logger.info(f"📊 Loading CSV: {self.file_path}")
        
        df = pd.read_csv(self.file_path)
        
        self._metadata = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns)
        }
        
        # Format as readable text
        self._content = f"CSV Data: {self.file_path.name}\n\n"
        self._content += f"Columns: {', '.join(df.columns)}\n"
        self._content += f"Rows: {len(df)}\n\n"
        self._content += df.to_string()
        
        logger.info(f"✅ Loaded CSV with {len(df)} rows, {len(df.columns)} columns")
        
        return self._content


@dataclass
class KnowledgeBase:
    """
    Collection of knowledge sources for agents.
    
    Example:
        kb = KnowledgeBase()
        await kb.add_pdf("research.pdf")
        await kb.add_web(["https://example.com"])
        
        content = kb.get_all_content()
    """
    sources: List[KnowledgeSource] = field(default_factory=list)
    _loaded: bool = False
    
    async def add_source(self, source: KnowledgeSource):
        """Add a knowledge source."""
        self.sources.append(source)
        logger.info(f"➕ Added {source.source_type.value} knowledge source")
    
    async def add_pdf(self, file_path: Union[str, Path], **kwargs):
        """Add PDF source."""
        source = PDFKnowledge(file_path, **kwargs)
        await self.add_source(source)
    
    async def add_web(self, urls: List[str], **kwargs):
        """Add web scraping source."""
        source = WebKnowledge(urls, **kwargs)
        await self.add_source(source)
    
    async def add_markdown(self, file_path: Union[str, Path], **kwargs):
        """Add markdown source."""
        source = MarkdownKnowledge(file_path, **kwargs)
        await self.add_source(source)
    
    async def add_text(self, content: Optional[str] = None, file_path: Optional[Union[str, Path]] = None, **kwargs):
        """Add text source."""
        source = TextKnowledge(content=content, file_path=file_path, **kwargs)
        await self.add_source(source)
    
    async def add_csv(self, file_path: Union[str, Path], **kwargs):
        """Add CSV source."""
        source = CSVKnowledge(file_path, **kwargs)
        await self.add_source(source)
    
    async def load_all(self):
        """Load all sources."""
        logger.info(f"📚 Loading {len(self.sources)} knowledge sources...")
        
        tasks = [source.load() for source in self.sources]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self._loaded = True
        logger.info("✅ All knowledge sources loaded")
    
    def get_all_content(self) -> str:
        """Get combined content from all sources."""
        if not self._loaded:
            raise RuntimeError("Knowledge base not loaded. Call load_all() first.")
        
        content = []
        for source in self.sources:
            content.append(f"\n{'='*60}")
            content.append(f"Source: {source.source_type.value}")
            if source.description:
                content.append(f"Description: {source.description}")
            content.append(f"{'='*60}\n")
            content.append(source.get_content())
        
        return "\n".join(content)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata from all sources."""
        return {
            "total_sources": len(self.sources),
            "sources": [
                {
                    "type": source.source_type.value,
                    "metadata": source.get_metadata()
                }
                for source in self.sources
            ]
        }
