"""
Data Hunters - Services for finding and downloading data

Hunters for different data sources:
- ArxivHunter: Search and download academic papers
- GoogleHunter: Search and download chart images from Google
- RoboflowHunter: Download pre-annotated datasets
"""

import hashlib
import json
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional

import requests
from loguru import logger

from ..config import (
    DataFactoryConfig,
    RAW_PDFS_DIR,
    SEARCH_CACHE_DIR,
    VERIFIED_ARXIV_IDS,
    ROBOFLOW_DATASETS,
)
from ..schemas import (
    ArxivPaper,
    DataSource,
    GoogleSearchResult,
    ProcessingStatus,
    RoboflowDataset,
)


# =============================================================================
# BASE HUNTER
# =============================================================================

class BaseHunter(ABC):
    """Abstract base class for data hunters."""
    
    def __init__(self, config: DataFactoryConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        })
    
    @abstractmethod
    def search(self, limit: int) -> List:
        """Search for data items."""
        pass
    
    @abstractmethod
    def download(self, item) -> bool:
        """Download a single item."""
        pass
    
    def _rate_limit(self, seconds: float) -> None:
        """Apply rate limiting."""
        jitter = random.uniform(0.5, 1.5)
        time.sleep(seconds * jitter)
    
    def _compute_hash(self, content: bytes) -> str:
        """Compute MD5 hash of content."""
        return hashlib.md5(content).hexdigest()[:12]


# =============================================================================
# ARXIV HUNTER
# =============================================================================

class ArxivHunter(BaseHunter):
    """
    Hunter for Arxiv academic papers.
    
    Searches Arxiv API for papers matching queries and downloads PDFs.
    """
    
    def __init__(self, config: DataFactoryConfig):
        super().__init__(config)
        self._ensure_arxiv_library()
    
    def _ensure_arxiv_library(self) -> None:
        """Ensure arxiv library is available."""
        try:
            import arxiv
            self.arxiv = arxiv
        except ImportError:
            logger.warning("arxiv library not installed, using direct API")
            self.arxiv = None
    
    def search(self, limit: int = 100) -> List[ArxivPaper]:
        """
        Search Arxiv for papers with charts.
        
        Args:
            limit: Maximum number of papers to return
            
        Returns:
            List of ArxivPaper objects
        """
        papers: List[ArxivPaper] = []
        seen_ids = set()
        
        # First, add verified papers (known to have good charts)
        logger.info(f"Adding {len(VERIFIED_ARXIV_IDS)} verified papers | source=arxiv")
        for arxiv_id in VERIFIED_ARXIV_IDS:
            if len(papers) >= limit:
                break
            if arxiv_id not in seen_ids:
                paper = self._fetch_paper_metadata(arxiv_id)
                if paper:
                    papers.append(paper)
                    seen_ids.add(arxiv_id)
        
        # Then search by queries
        results_per_query = max(1, (limit - len(papers)) // len(self.config.arxiv_queries))
        
        for query in self.config.arxiv_queries:
            if len(papers) >= limit:
                break
                
            logger.info(f"Searching Arxiv | query='{query[:50]}...' | limit={results_per_query}")
            
            try:
                query_results = self._search_arxiv(query, results_per_query)
                for paper in query_results:
                    if paper.arxiv_id not in seen_ids:
                        paper.search_query = query
                        papers.append(paper)
                        seen_ids.add(paper.arxiv_id)
                        
                        if len(papers) >= limit:
                            break
                
                self._rate_limit(self.config.arxiv_rate_limit_seconds)
                
            except Exception as e:
                logger.error(f"Search failed | query='{query}' | error={e}")
                continue
        
        logger.info(f"Arxiv search complete | total_papers={len(papers)}")
        return papers[:limit]
    
    def _search_arxiv(self, query: str, max_results: int) -> List[ArxivPaper]:
        """Execute Arxiv search."""
        papers = []
        
        # Limit per query to avoid timeout (Arxiv recommends max 100)
        max_per_query = min(max_results, 100)
        
        if self.arxiv:
            # Use arxiv library with proper client settings
            client = self.arxiv.Client(
                page_size=50,
                delay_seconds=3.0,
                num_retries=3,
            )
            
            search = self.arxiv.Search(
                query=query,
                max_results=max_per_query,
                sort_by=self.arxiv.SortCriterion.SubmittedDate,
            )
            
            try:
                for result in client.results(search):
                    paper = ArxivPaper(
                        arxiv_id=result.entry_id.split("/")[-1],
                        title=result.title,
                        authors=[a.name for a in result.authors],
                        abstract=result.summary,
                        published_date=result.published,
                        pdf_url=result.pdf_url,
                        categories=result.categories,
                    )
                    papers.append(paper)
            except Exception as e:
                logger.warning(f"Arxiv search partial failure | query={query[:30]} | error={e}")
        else:
            # Direct API call
            papers = self._search_arxiv_api(query, max_per_query)
        
        return papers
    
    def _search_arxiv_api(self, query: str, max_results: int) -> List[ArxivPaper]:
        """Search using Arxiv API directly."""
        import xml.etree.ElementTree as ET
        
        base_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        
        response = self.session.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        
        papers = []
        for entry in root.findall("atom:entry", ns):
            arxiv_id = entry.find("atom:id", ns).text.split("/")[-1]
            title = entry.find("atom:title", ns).text.strip()
            published = entry.find("atom:published", ns).text
            
            # Find PDF link
            pdf_url = None
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href")
                    break
            
            if pdf_url:
                papers.append(ArxivPaper(
                    arxiv_id=arxiv_id,
                    title=title,
                    published_date=datetime.fromisoformat(published.replace("Z", "+00:00")),
                    pdf_url=pdf_url,
                ))
        
        return papers
    
    def _fetch_paper_metadata(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """Fetch metadata for a specific paper."""
        try:
            papers = self._search_arxiv_api(f"id:{arxiv_id}", 1)
            if papers:
                return papers[0]
        except Exception as e:
            logger.warning(f"Failed to fetch metadata | arxiv_id={arxiv_id} | error={e}")
        
        # Fallback: create minimal paper object
        return ArxivPaper(
            arxiv_id=arxiv_id,
            title=f"Paper {arxiv_id}",
            published_date=datetime.now(),
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        )
    
    def download(self, paper: ArxivPaper) -> bool:
        """
        Download PDF for a paper.
        
        Args:
            paper: ArxivPaper object
            
        Returns:
            True if download successful
        """
        save_path = RAW_PDFS_DIR / f"arxiv_{paper.safe_id}.pdf"
        
        # Skip if already exists
        if save_path.exists():
            logger.debug(f"PDF exists, skipping | arxiv_id={paper.arxiv_id}")
            paper.local_pdf_path = save_path
            paper.status = ProcessingStatus.DOWNLOADED
            return True
        
        logger.info(f"Downloading PDF | arxiv_id={paper.arxiv_id} | url={paper.pdf_url}")
        
        try:
            response = self.session.get(
                paper.pdf_url,
                timeout=self.config.request_timeout,
                stream=True,
            )
            response.raise_for_status()
            
            # Verify it's a PDF
            content_type = response.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower():
                # Check magic bytes
                first_bytes = response.content[:5]
                if first_bytes != b"%PDF-":
                    logger.warning(f"Not a valid PDF | arxiv_id={paper.arxiv_id}")
                    paper.status = ProcessingStatus.FAILED
                    return False
            
            # Save file
            RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            paper.local_pdf_path = save_path
            paper.status = ProcessingStatus.DOWNLOADED
            logger.info(f"Downloaded | arxiv_id={paper.arxiv_id} | path={save_path}")
            
            self._rate_limit(self.config.arxiv_rate_limit_seconds)
            return True
            
        except Exception as e:
            logger.error(f"Download failed | arxiv_id={paper.arxiv_id} | error={e}")
            paper.status = ProcessingStatus.FAILED
            return False
    
    def download_all(self, papers: List[ArxivPaper]) -> int:
        """
        Download all papers.
        
        Returns:
            Number of successfully downloaded papers
        """
        success_count = 0
        for i, paper in enumerate(papers):
            logger.info(f"Progress | {i+1}/{len(papers)} | arxiv_id={paper.arxiv_id}")
            if self.download(paper):
                success_count += 1
        
        logger.info(f"Download complete | success={success_count}/{len(papers)}")
        return success_count


# =============================================================================
# GOOGLE HUNTER
# =============================================================================

class GoogleHunter(BaseHunter):
    """
    Hunter for Google Image Search results.
    
    Uses SerpAPI for reliable Google search, falls back to direct scraping.
    """
    
    def __init__(self, config: DataFactoryConfig):
        super().__init__(config)
        self.serpapi_key = config.serpapi_key
        self.cache_dir = SEARCH_CACHE_DIR / "google"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def search(self, limit: int = 100) -> List[GoogleSearchResult]:
        """
        Search Google for chart images.
        
        Args:
            limit: Maximum number of images
            
        Returns:
            List of GoogleSearchResult objects
        """
        results: List[GoogleSearchResult] = []
        seen_urls = set()
        
        results_per_query = max(1, limit // len(self.config.google_queries))
        
        for query in self.config.google_queries:
            if len(results) >= limit:
                break
            
            logger.info(f"Searching Google Images | query='{query}' | limit={results_per_query}")
            
            try:
                # Check cache first
                cache_file = self.cache_dir / f"{self._compute_hash(query.encode())}.json"
                if cache_file.exists():
                    with open(cache_file) as f:
                        cached = json.load(f)
                        for item in cached:
                            if item["image_url"] not in seen_urls:
                                results.append(GoogleSearchResult(**item))
                                seen_urls.add(item["image_url"])
                    continue
                
                # Search
                query_results = self._search_google(query, results_per_query)
                
                # Cache results
                with open(cache_file, "w") as f:
                    json.dump([r.model_dump(mode="json") for r in query_results], f)
                
                for result in query_results:
                    if result.image_url not in seen_urls:
                        results.append(result)
                        seen_urls.add(result.image_url)
                
                self._rate_limit(2.0)
                
            except Exception as e:
                logger.error(f"Google search failed | query='{query}' | error={e}")
                continue
        
        logger.info(f"Google search complete | total_results={len(results)}")
        return results[:limit]
    
    def _search_google(self, query: str, num_results: int) -> List[GoogleSearchResult]:
        """Execute Google image search."""
        if self.serpapi_key:
            return self._search_serpapi(query, num_results)
        else:
            logger.warning("No SerpAPI key, using limited fallback method")
            return self._search_fallback(query, num_results)
    
    def _search_serpapi(self, query: str, num_results: int) -> List[GoogleSearchResult]:
        """Search using SerpAPI."""
        try:
            from serpapi import GoogleSearch
        except ImportError:
            logger.error("serpapi not installed: pip install google-search-results")
            return []
        
        params = {
            "q": query,
            "tbm": "isch",  # Image search
            "num": num_results,
            "api_key": self.serpapi_key,
        }
        
        search = GoogleSearch(params)
        data = search.get_dict()
        
        results = []
        for item in data.get("images_results", []):
            results.append(GoogleSearchResult(
                query=query,
                image_url=item.get("original"),
                thumbnail_url=item.get("thumbnail"),
                source_page_url=item.get("link"),
                title=item.get("title"),
            ))
        
        return results
    
    def _search_fallback(self, query: str, num_results: int) -> List[GoogleSearchResult]:
        """
        Fallback search without API.
        
        Note: This is limited and may not work reliably.
        Consider getting a free SerpAPI key for better results.
        """
        # For now, return empty - encourage users to get API key
        logger.info("Fallback search returns limited results. Get SerpAPI key for full access.")
        return []
    
    def download(self, result: GoogleSearchResult) -> bool:
        """
        Download an image from search result.
        
        Args:
            result: GoogleSearchResult object
            
        Returns:
            True if download successful
        """
        if not result.image_url:
            return False
        
        # Generate filename from URL hash
        url_hash = self._compute_hash(result.image_url.encode())
        save_path = self.config.images_dir / f"google_{url_hash}.png"
        
        if save_path.exists():
            result.local_path = save_path
            result.status = ProcessingStatus.DOWNLOADED
            return True
        
        try:
            response = self.session.get(
                result.image_url,
                timeout=self.config.request_timeout,
            )
            response.raise_for_status()
            
            # Save image
            with open(save_path, "wb") as f:
                f.write(response.content)
            
            result.local_path = save_path
            result.status = ProcessingStatus.DOWNLOADED
            
            self._rate_limit(1.0)
            return True
            
        except Exception as e:
            logger.warning(f"Image download failed | url={result.image_url[:50]}... | error={e}")
            result.status = ProcessingStatus.FAILED
            return False


# =============================================================================
# ROBOFLOW HUNTER
# =============================================================================

class RoboflowHunter(BaseHunter):
    """
    Hunter for Roboflow pre-annotated datasets.
    
    Downloads complete datasets with annotations in YOLO format.
    """
    
    def __init__(self, config: DataFactoryConfig, api_key: Optional[str] = None):
        super().__init__(config)
        self.api_key = api_key or "public"
    
    def search(self, limit: int = 100) -> List[RoboflowDataset]:
        """
        List available Roboflow datasets.
        
        Returns:
            List of RoboflowDataset objects
        """
        datasets = []
        for name, info in ROBOFLOW_DATASETS.items():
            datasets.append(RoboflowDataset(
                name=name,
                workspace=info["workspace"],
                project=info["project"],
                version=info["version"],
                description=info["description"],
                classes=info["classes"],
            ))
        return datasets
    
    def download(self, dataset: RoboflowDataset) -> bool:
        """
        Download a Roboflow dataset.
        
        Args:
            dataset: RoboflowDataset object
            
        Returns:
            True if download successful
        """
        try:
            from roboflow import Roboflow
        except ImportError:
            logger.error("roboflow not installed: pip install roboflow")
            return False
        
        output_dir = self.config.data_dir / "training" / dataset.name
        
        if output_dir.exists() and any(output_dir.iterdir()):
            logger.info(f"Dataset exists | name={dataset.name} | path={output_dir}")
            dataset.local_path = output_dir
            dataset.status = ProcessingStatus.DOWNLOADED
            return True
        
        logger.info(f"Downloading Roboflow dataset | name={dataset.name}")
        
        try:
            rf = Roboflow(api_key=self.api_key)
            project = rf.workspace(dataset.workspace).project(dataset.project)
            
            project.version(dataset.version).download(
                "yolov8",
                location=str(output_dir),
            )
            
            dataset.local_path = output_dir
            dataset.status = ProcessingStatus.DOWNLOADED
            
            # Count images
            images_dir = output_dir / "train" / "images"
            if images_dir.exists():
                dataset.images_count = len(list(images_dir.glob("*")))
            
            logger.info(f"Downloaded | name={dataset.name} | images={dataset.images_count}")
            return True
            
        except Exception as e:
            logger.error(f"Roboflow download failed | name={dataset.name} | error={e}")
            dataset.status = ProcessingStatus.FAILED
            return False
