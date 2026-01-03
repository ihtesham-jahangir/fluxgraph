"""
FluxGraph Data Parser

Parse and extract structured data from various formats:
- JSON data
- CSV/TSV data
- XML data
- YAML data
- Table extraction
- List extraction
- Key-value extraction
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)


class DataParser:
    """
    Universal data parser

    Example:
        parser = DataParser()

        # Parse JSON
        data = parser.parse_json('{"name": "John", "age": 30}')

        # Extract emails
        emails = parser.extract_emails(text)

        # Extract URLs
        urls = parser.extract_urls(text)

        # Parse tables
        tables = parser.extract_tables(markdown_text)
    """

    # Common regex patterns
    PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "url": r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*',
        "phone_us": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "phone_intl": r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
        "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        "date_iso": r'\b\d{4}-\d{2}-\d{2}\b',
        "price": r'\$\s?\d+(?:,\d{3})*(?:\.\d{2})?',
        "hashtag": r'#\w+',
        "mention": r'@\w+',
        "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
    }

    def parse_json(self, text: str, strict: bool = True) -> Union[Dict, List, None]:
        """
        Parse JSON string

        Args:
            text: JSON string
            strict: Use strict parsing

        Returns:
            Parsed JSON data
        """
        try:
            return json.loads(text, strict=strict)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return None

    def parse_csv(
        self,
        text: str,
        delimiter: str = ",",
        has_header: bool = True
    ) -> List[Dict]:
        """
        Parse CSV text

        Args:
            text: CSV text
            delimiter: Column delimiter
            has_header: First row is header

        Returns:
            List of dictionaries
        """
        import csv
        from io import StringIO

        reader = csv.reader(StringIO(text), delimiter=delimiter)
        rows = list(reader)

        if not rows:
            return []

        if has_header and len(rows) > 1:
            headers = rows[0]
            return [dict(zip(headers, row)) for row in rows[1:]]
        else:
            return [{"col_" + str(i): val for i, val in enumerate(row)} for row in rows]

    def parse_yaml(self, text: str) -> Any:
        """Parse YAML text"""
        try:
            import yaml
            return yaml.safe_load(text)
        except ImportError:
            raise ImportError("PyYAML required. Install with: pip install pyyaml")
        except Exception as e:
            logger.error(f"YAML parse error: {e}")
            return None

    def parse_xml(self, text: str) -> Dict:
        """
        Parse XML to dictionary

        Returns:
            Dictionary representation of XML
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 required. Install with: pip install beautifulsoup4")

        soup = BeautifulSoup(text, 'xml') or BeautifulSoup(text, 'html.parser')
        return self._xml_to_dict(soup)

    def _xml_to_dict(self, element) -> Dict:
        """Convert XML element to dictionary"""
        result = {}

        # Get element attributes
        if hasattr(element, 'attrs'):
            result.update({"@" + k: v for k, v in element.attrs.items()})

        # Get child elements
        children = list(element.children) if hasattr(element, 'children') else []
        text_content = []

        for child in children:
            if hasattr(child, 'name') and child.name:
                child_dict = self._xml_to_dict(child)
                if child.name in result:
                    # Convert to list if multiple children with same name
                    if not isinstance(result[child.name], list):
                        result[child.name] = [result[child.name]]
                    result[child.name].append(child_dict)
                else:
                    result[child.name] = child_dict
            elif isinstance(child, str) and child.strip():
                text_content.append(child.strip())

        if text_content and not result:
            return " ".join(text_content)
        elif text_content:
            result["#text"] = " ".join(text_content)

        return result

    def extract_emails(self, text: str) -> List[str]:
        """Extract all email addresses"""
        return re.findall(self.PATTERNS["email"], text)

    def extract_urls(self, text: str) -> List[str]:
        """Extract all URLs"""
        return re.findall(self.PATTERNS["url"], text)

    def extract_phone_numbers(self, text: str, format: str = "us") -> List[str]:
        """
        Extract phone numbers

        Args:
            text: Text to search
            format: "us" or "intl"
        """
        pattern = self.PATTERNS["phone_us"] if format == "us" else self.PATTERNS["phone_intl"]
        return re.findall(pattern, text)

    def extract_dates(self, text: str) -> List[str]:
        """Extract ISO format dates (YYYY-MM-DD)"""
        return re.findall(self.PATTERNS["date_iso"], text)

    def extract_prices(self, text: str) -> List[str]:
        """Extract dollar amounts"""
        return re.findall(self.PATTERNS["price"], text)

    def extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags"""
        return re.findall(self.PATTERNS["hashtag"], text)

    def extract_mentions(self, text: str) -> List[str]:
        """Extract @mentions"""
        return re.findall(self.PATTERNS["mention"], text)

    def extract_ip_addresses(self, text: str) -> List[str]:
        """Extract IP addresses"""
        return re.findall(self.PATTERNS["ip_address"], text)

    def extract_tables(self, markdown_text: str) -> List[List[List[str]]]:
        """
        Extract tables from markdown text

        Returns:
            List of tables (each table is a list of rows)
        """
        tables = []
        lines = markdown_text.split('\n')

        current_table = []
        in_table = False

        for line in lines:
            # Check if line is a table row (contains |)
            if '|' in line:
                # Skip separator rows (----)
                if re.match(r'^[\s|:-]+$', line):
                    continue

                cells = [cell.strip() for cell in line.split('|')]
                # Remove empty cells from start/end
                cells = [c for c in cells if c]

                if cells:
                    current_table.append(cells)
                    in_table = True
            elif in_table and current_table:
                # End of table
                tables.append(current_table)
                current_table = []
                in_table = False

        # Add last table if exists
        if current_table:
            tables.append(current_table)

        return tables

    def extract_lists(self, text: str) -> Dict[str, List[str]]:
        """
        Extract bullet and numbered lists

        Returns:
            Dictionary with 'bullet' and 'numbered' lists
        """
        bullet_pattern = r'^\s*[-*+]\s+(.+)$'
        numbered_pattern = r'^\s*\d+\.\s+(.+)$'

        bullets = []
        numbered = []

        for line in text.split('\n'):
            bullet_match = re.match(bullet_pattern, line)
            if bullet_match:
                bullets.append(bullet_match.group(1))

            numbered_match = re.match(numbered_pattern, line)
            if numbered_match:
                numbered.append(numbered_match.group(1))

        return {
            "bullet": bullets,
            "numbered": numbered
        }

    def extract_key_values(
        self,
        text: str,
        separators: List[str] = None
    ) -> Dict[str, str]:
        """
        Extract key-value pairs

        Args:
            text: Text containing key-value pairs
            separators: List of separators (default: [':', '=', '->'])

        Returns:
            Dictionary of key-value pairs
        """
        separators = separators or [':', '=', '->']

        result = {}
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue

            for sep in separators:
                if sep in line:
                    parts = line.split(sep, 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        result[key] = value
                    break

        return result

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text

        - Remove extra whitespace
        - Remove special characters (optional)
        - Normalize line endings
        """
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)

        # Remove multiple newlines
        text = re.sub(r'\n\n+', '\n\n', text)

        return text.strip()

    def extract_code_blocks(self, markdown_text: str) -> List[Dict[str, str]]:
        """
        Extract code blocks from markdown

        Returns:
            List of dictionaries with 'language' and 'code'
        """
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, markdown_text, re.DOTALL)

        return [
            {"language": lang or "text", "code": code.strip()}
            for lang, code in matches
        ]

    def tokenize(self, text: str, method: str = "words") -> List[str]:
        """
        Tokenize text

        Args:
            text: Text to tokenize
            method: "words", "sentences", or "paragraphs"

        Returns:
            List of tokens
        """
        if method == "words":
            return re.findall(r'\b\w+\b', text.lower())

        elif method == "sentences":
            return re.split(r'[.!?]+', text)

        elif method == "paragraphs":
            return [p.strip() for p in text.split('\n\n') if p.strip()]

        else:
            raise ValueError(f"Unknown tokenization method: {method}")

    def extract_numbers(self, text: str) -> List[float]:
        """Extract all numbers from text"""
        # Match integers and decimals
        pattern = r'-?\d+\.?\d*'
        matches = re.findall(pattern, text)
        return [float(m) for m in matches if m]


# Convenience functions

def parse_json_str(text: str) -> Union[Dict, List, None]:
    """Quick JSON parsing"""
    parser = DataParser()
    return parser.parse_json(text)


def extract_all_data(text: str) -> Dict[str, List]:
    """
    Extract all common data types from text

    Returns:
        Dictionary with extracted emails, URLs, phones, etc.
    """
    parser = DataParser()

    return {
        "emails": parser.extract_emails(text),
        "urls": parser.extract_urls(text),
        "phone_numbers": parser.extract_phone_numbers(text),
        "dates": parser.extract_dates(text),
        "prices": parser.extract_prices(text),
        "hashtags": parser.extract_hashtags(text),
        "mentions": parser.extract_mentions(text),
        "ip_addresses": parser.extract_ip_addresses(text)
    }
