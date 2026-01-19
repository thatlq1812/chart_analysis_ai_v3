"""
ACL Anthology Hunter

Download papers from ACL Anthology (NLP/AI conferences).
Excellent source for:
- Model comparison charts (bar, line)
- Performance benchmark visualizations
- Clear, academic-style figures

ACL Anthology is a static website - easy to scrape, no rate limiting issues.
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from loguru import logger

from ..config import (
    DataFactoryConfig,
    RAW_PDFS_DIR,
    SEARCH_CACHE_DIR,
)
from ..schemas import (
    ArxivPaper,
    DataSource,
    ProcessingStatus,
)


# =============================================================================
# ACL CONFIGURATION
# =============================================================================

ACL_BASE_URL = "https://aclanthology.org"

# High-quality venues with lots of charts
ACL_VENUES = {
    "acl": {
        "name": "ACL",
        "description": "Annual Meeting of the Association for Computational Linguistics",
        "years": list(range(2020, 2026)),
    },
    "emnlp": {
        "name": "EMNLP",
        "description": "Empirical Methods in Natural Language Processing",
        "years": list(range(2020, 2026)),
    },
    "naacl": {
        "name": "NAACL",
        "description": "North American Chapter of ACL",
        "years": list(range(2020, 2026)),
    },
    "findings": {
        "name": "Findings",
        "description": "Findings of ACL/EMNLP",
        "years": list(range(2021, 2026)),
    },
    "eacl": {
        "name": "EACL",
        "description": "European Chapter of ACL",
        "years": [2021, 2023, 2024],
    },
}

# Search terms for papers with charts
CHART_KEYWORDS = [
    "benchmark",
    "comparison",
    "evaluation",
    "performance",
    "ablation",
    "analysis",
    "visualization",
]


# =============================================================================
# ACL HUNTER
# =============================================================================

class ACLHunter:
    """
    Hunter for ACL Anthology papers.
    
    Scrapes the ACL Anthology website to find and download
    NLP papers with performance charts.
    
    Example:
        hunter = ACLHunter(config)
        papers = hunter.search(venue="acl", year=2023, limit=50)
        for paper in papers:
            hunter.download_pdf(paper)
    """
    
    def __init__(self, config: DataFactoryConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; GeoSLM-Research/1.0)"
        })
        
        # Gentle rate limiting for static site
        self.request_delay = 1.0
    
    def search(
        self,
        venue: Optional[str] = None,
        year: Optional[int] = None,
        keyword: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Search ACL Anthology for papers.
        
        Args:
            venue: Conference venue (acl, emnlp, naacl, etc.)
            year: Publication year
            keyword: Search keyword in title/abstract
            limit: Maximum number of papers
            
        Returns:
            List of paper metadata dicts
        """
        papers = []
        
        if venue and year:
            # Search specific venue and year
            papers = self._search_venue_year(venue, year, limit)
        elif venue:
            # Search all years of a venue
            venue_info = ACL_VENUES.get(venue, {})
            years = venue_info.get("years", [2023])
            
            for y in years:
                papers.extend(self._search_venue_year(venue, y, limit // len(years)))
                if len(papers) >= limit:
                    break
        else:
            # Search across multiple venues
            for v in ["acl", "emnlp"]:
                for y in [2023, 2024, 2025]:
                    papers.extend(self._search_venue_year(v, y, limit // 6))
                    if len(papers) >= limit:
                        break
        
        # Filter by keyword if specified
        if keyword:
            papers = [
                p for p in papers
                if keyword.lower() in p.get("title", "").lower()
                or keyword.lower() in p.get("abstract", "").lower()
            ]
        
        return papers[:limit]
    
    def _search_venue_year(
        self,
        venue: str,
        year: int,
        limit: int,
    ) -> List[Dict]:
        """Search a specific venue and year."""
        logger.info(f"Searching ACL | venue={venue} | year={year}")
        
        # ACL Anthology URL structure
        url = f"{ACL_BASE_URL}/events/{venue}-{year}/"
        
        try:
            response = self.session.get(url, timeout=30)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch | url={url} | status={response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, "html.parser")
            papers = []
            
            # Find paper entries
            paper_entries = soup.find_all("p", class_="d-sm-flex")
            
            for entry in paper_entries[:limit]:
                paper = self._parse_paper_entry(entry, venue, year)
                if paper:
                    papers.append(paper)
            
            logger.info(f"Found {len(papers)} papers | venue={venue} | year={year}")
            
            time.sleep(self.request_delay)
            return papers
            
        except Exception as e:
            logger.error(f"Failed to search | venue={venue} | year={year} | error={e}")
            return []
    
    def _parse_paper_entry(
        self,
        entry,
        venue: str,
        year: int,
    ) -> Optional[Dict]:
        """Parse a paper entry from the page."""
        try:
            # Get title and link
            title_elem = entry.find("strong")
            if not title_elem:
                return None
            
            title_link = title_elem.find("a")
            if not title_link:
                return None
            
            title = title_link.text.strip()
            paper_url = urljoin(ACL_BASE_URL, title_link.get("href", ""))
            
            # Extract paper ID from URL
            paper_id = paper_url.split("/")[-1].rstrip("/")
            
            # Get PDF link
            pdf_link = entry.find("a", class_="badge-primary")
            pdf_url = None
            if pdf_link and "pdf" in pdf_link.text.lower():
                pdf_url = urljoin(ACL_BASE_URL, pdf_link.get("href", ""))
            
            # Get authors
            authors = []
            author_links = entry.find_all("a", href=lambda x: x and "/people/" in x)
            for author in author_links:
                authors.append(author.text.strip())
            
            return {
                "paper_id": paper_id,
                "title": title,
                "authors": authors,
                "venue": venue,
                "year": year,
                "paper_url": paper_url,
                "pdf_url": pdf_url,
                "source": "acl_anthology",
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse paper entry | error={e}")
            return None
    
    def download_pdf(
        self,
        paper: Dict,
        output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Download PDF for a paper.
        
        Args:
            paper: Paper metadata dict
            output_dir: Output directory
            
        Returns:
            Path to downloaded PDF or None
        """
        if output_dir is None:
            output_dir = RAW_PDFS_DIR / "acl"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_url = paper.get("pdf_url")
        if not pdf_url:
            logger.warning(f"No PDF URL | paper_id={paper.get('paper_id')}")
            return None
        
        paper_id = paper.get("paper_id", "unknown")
        output_path = output_dir / f"{paper_id}.pdf"
        
        # Skip if already downloaded
        if output_path.exists():
            logger.debug(f"Already downloaded | paper_id={paper_id}")
            return output_path
        
        logger.info(f"Downloading PDF | paper_id={paper_id}")
        
        try:
            response = self.session.get(pdf_url, timeout=60, stream=True)
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Save metadata
            meta_path = output_path.with_suffix(".json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(paper, f, indent=2, ensure_ascii=False)
            
            time.sleep(self.request_delay)
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to download | paper_id={paper_id} | error={e}")
            return None
    
    def hunt(
        self,
        limit: int = 100,
        venues: Optional[List[str]] = None,
    ) -> Generator[ArxivPaper, None, None]:
        """
        Complete hunting workflow: search + download.
        
        Args:
            limit: Total number of papers to download
            venues: List of venues to search (default: ["acl", "emnlp"])
            
        Yields:
            ArxivPaper-like objects (reusing schema)
        """
        if venues is None:
            venues = ["acl", "emnlp"]
        
        logger.info(f"Starting ACL hunt | target={limit} | venues={venues}")
        
        # Search for papers
        papers = []
        per_venue = limit // len(venues)
        
        for venue in venues:
            venue_papers = self.search(venue=venue, limit=per_venue)
            papers.extend(venue_papers)
        
        if not papers:
            logger.warning("No papers found")
            return
        
        logger.info(f"Found {len(papers)} papers, downloading...")
        
        # Download PDFs
        count = 0
        for paper in papers:
            if count >= limit:
                break
            
            pdf_path = self.download_pdf(paper)
            
            if pdf_path:
                # Convert to ArxivPaper-like format for compatibility
                arxiv_paper = ArxivPaper(
                    arxiv_id=f"acl_{paper['paper_id']}",
                    title=paper["title"],
                    authors=paper.get("authors", []),
                    abstract="",  # ACL doesn't provide abstract in listing
                    published_date=datetime(paper["year"], 1, 1),
                    pdf_url=paper.get("pdf_url", ""),
                    local_pdf_path=pdf_path,
                    status=ProcessingStatus.DOWNLOADED,
                    categories=[paper["venue"]],
                    search_query=f"acl_{paper['venue']}_{paper['year']}",
                )
                
                count += 1
                yield arxiv_paper
        
        logger.success(f"ACL hunt complete | downloaded={count}")
    
    def search_with_charts(self, limit: int = 50) -> List[Dict]:
        """
        Search specifically for papers likely to have charts.
        
        Uses keywords that correlate with papers having
        performance comparison charts.
        """
        logger.info("Searching for chart-rich papers...")
        
        all_papers = []
        
        # Search recent high-impact venues
        for venue in ["acl", "emnlp"]:
            for year in [2024, 2023]:
                papers = self.search(venue=venue, year=year, limit=100)
                
                # Filter for chart-likely papers
                chart_papers = [
                    p for p in papers
                    if any(kw in p.get("title", "").lower() for kw in CHART_KEYWORDS)
                ]
                
                all_papers.extend(chart_papers)
                
                if len(all_papers) >= limit:
                    break
        
        logger.info(f"Found {len(all_papers)} chart-likely papers")
        return all_papers[:limit]
