
import re
import requests
from urllib.parse import urlparse
from ipaddress import ip_address, ip_network
from bs4 import BeautifulSoup
from typing import Optional, List, Dict
from core.logger import get_logger
from core.config import QDRANT_RESEARCH_COLLECTION

logger = get_logger(__name__)

# Private IP ranges that should be blocked to prevent SSRF attacks
BLOCKED_NETWORKS = [
    ip_network("10.0.0.0/8"),
    ip_network("172.16.0.0/12"),
    ip_network("192.168.0.0/16"),
    ip_network("127.0.0.0/8"),
    ip_network("169.254.0.0/16"),
    ip_network("0.0.0.0/8"),
]


class WebIngestor:
    """Fetches, cleans, chunks, and optionally stores web page content into Qdrant.
    
    Safety:
        - Blocks requests to private/internal IP ranges (SSRF protection).
        - Enforces a request timeout (default 15 seconds).
        - Uses a descriptive User-Agent header.
    """

    DEFAULT_CHUNK_SIZE = 800
    REQUEST_TIMEOUT = 15

    def __init__(self, retriever=None):
        """
        Args:
            retriever: Optional Retriever instance for Qdrant storage.
                       If provided, ``ingest_url`` will persist chunks to the
                       research collection automatically.
        """
        self.retriever = retriever

    # ── Safety ──────────────────────────────────────────────────────────

    @staticmethod
    def _is_safe_url(url: str) -> bool:
        """Validates URL is not targeting a private/internal network."""
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                return False

            # Resolve hostname to check IP
            import socket
            resolved_ip = socket.gethostbyname(hostname)
            addr = ip_address(resolved_ip)
            for network in BLOCKED_NETWORKS:
                if addr in network:
                    logger.warning(f"Blocked SSRF attempt to private IP: {url} → {resolved_ip}")
                    return False
            return True
        except Exception as e:
            logger.error(f"URL safety check failed for {url}: {e}")
            return False

    # ── Fetching ────────────────────────────────────────────────────────

    def fetch_page(self, url: str) -> Optional[str]:
        """Fetches raw HTML from a URL with safety checks."""
        if not self._is_safe_url(url):
            logger.error(f"URL blocked by safety policy: {url}")
            return None

        try:
            headers = {"User-Agent": "ENML-Research-Bot/2.0 (+https://github.com/flexcreates/ENML)"}
            response = requests.get(url, headers=headers, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    # ── Content Extraction ──────────────────────────────────────────────

    def extract_content(self, html: str) -> str:
        """Strips HTML to clean readable text."""
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Remove non-content elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = soup.get_text()

            # Clean whitespace
            lines = []
            for line in text.splitlines():
                clean_line = re.sub(r'\s+', ' ', line).strip()
                if clean_line and len(clean_line) > 3:
                    lines.append(clean_line)

            return '\n'.join(lines)
        except Exception as e:
            logger.error(f"Failed to extract content: {e}")
            return ""

    # ── Chunking ────────────────────────────────────────────────────────

    def chunk_text(self, text: str, chunk_size: int = None) -> List[str]:
        """Splits text into overlapping chunks for embedding.
        
        Uses a 10% overlap between consecutive chunks to preserve context
        across chunk boundaries.
        """
        chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
        overlap = chunk_size // 10
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    # ── Full Pipeline ───────────────────────────────────────────────────

    def ingest_url(self, url: str, topic: str = "general") -> Optional[List[str]]:
        """Full pipeline: fetch → clean → chunk → (optionally) store in Qdrant.
        
        Args:
            url: The web page URL to ingest.
            topic: Topic label for metadata tagging.
            
        Returns:
            List of text chunks, or None on failure.
        """
        html = self.fetch_page(url)
        if not html:
            return None

        text = self.extract_content(html)
        if not text:
            logger.warning(f"No usable content extracted from {url}")
            return None

        chunks = self.chunk_text(text)
        logger.info(f"Extracted {len(chunks)} chunks ({len(text)} chars) from {url}")

        # Persist to Qdrant if retriever is available
        if self.retriever:
            for i, chunk in enumerate(chunks):
                payload = {
                    "type": "research",
                    "source_url": url,
                    "topic": topic,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
                self.retriever.add_memory(QDRANT_RESEARCH_COLLECTION, chunk, payload)
            logger.info(f"Stored {len(chunks)} chunks from {url} into Qdrant")

        return chunks
