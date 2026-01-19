"""
PubMed Central (PMC) Hunter

Download open-access biomedical papers with charts.
PMC is excellent for:
- Statistical charts (bar, box plots, histograms)
- High-quality peer-reviewed figures
- Detailed captions with statistical info

Uses NCBI Entrez API (official, no rate limit issues).
"""

import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional
from urllib.parse import urlencode

import requests
from loguru import logger

from ..config import (
    DataFactoryConfig,
    RAW_PDFS_DIR,
    IMAGES_DIR,
    METADATA_DIR,
    SEARCH_CACHE_DIR,
)
from ..schemas import (
    ChartImage,
    ChartType,
    DataSource,
    ProcessingStatus,
)


# =============================================================================
# PMC CONFIGURATION
# =============================================================================

PMC_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_FTP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc"

# Search queries for chart-rich papers
PMC_CHART_QUERIES = [
    # Statistical analysis papers (lots of bar charts, box plots)
    '("bar chart"[Body - All Words]) AND open access[filter]',
    '("line graph"[Body - All Words]) AND open access[filter]',
    '("pie chart"[Body - All Words]) AND open access[filter]',
    '("scatter plot"[Body - All Words]) AND open access[filter]',
    '("box plot"[Body - All Words]) AND open access[filter]',
    
    # Clinical trials (performance comparison charts)
    '("comparison chart"[Body - All Words]) AND clinical trial[pt]',
    
    # Bioinformatics (lots of visualizations)
    '(visualization[Title]) AND bioinformatics[Journal]',
]


# =============================================================================
# PMC HUNTER
# =============================================================================

class PMCHunter:
    """
    Hunter for PubMed Central open-access papers.
    
    Uses NCBI Entrez API to:
    1. Search for papers with charts
    2. Download full-text XML/PDF
    3. Extract figure images with captions
    
    Example:
        hunter = PMCHunter(config)
        papers = hunter.search(limit=50)
        for paper in papers:
            hunter.download_figures(paper)
    """
    
    def __init__(self, config: DataFactoryConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "GeoSLM-ChartAnalysis/1.0 (Academic Research)"
        })
        
        # API key (optional but increases rate limit)
        self.api_key = config.pmc_api_key if hasattr(config, 'pmc_api_key') else None
        
        # Rate limiting: 3 requests/second without API key
        self.request_delay = 0.34 if not self.api_key else 0.1
    
    def search(
        self,
        query: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Search PMC for papers matching query.
        
        Args:
            query: Search query (uses default chart queries if None)
            limit: Maximum number of papers to return
            
        Returns:
            List of paper metadata dicts
        """
        if query is None:
            # Use default chart-finding queries
            all_results = []
            per_query_limit = max(10, limit // len(PMC_CHART_QUERIES))
            
            for q in PMC_CHART_QUERIES:
                results = self._search_query(q, per_query_limit)
                all_results.extend(results)
                
                if len(all_results) >= limit:
                    break
            
            return all_results[:limit]
        
        return self._search_query(query, limit)
    
    def _search_query(self, query: str, limit: int) -> List[Dict]:
        """Execute a single search query."""
        logger.info(f"Searching PMC | query={query[:50]}... | limit={limit}")
        
        # Step 1: ESearch to get PMCIDs
        search_params = {
            "db": "pmc",
            "term": query,
            "retmax": limit,
            "retmode": "json",
            "usehistory": "y",
        }
        
        if self.api_key:
            search_params["api_key"] = self.api_key
        
        search_url = f"{PMC_BASE_URL}/esearch.fcgi?{urlencode(search_params)}"
        
        try:
            response = self.session.get(search_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            id_list = data.get("esearchresult", {}).get("idlist", [])
            
            if not id_list:
                logger.warning(f"No results for query: {query[:50]}...")
                return []
            
            logger.info(f"Found {len(id_list)} papers")
            
            time.sleep(self.request_delay)
            
            # Step 2: EFetch to get metadata
            return self._fetch_metadata(id_list)
            
        except Exception as e:
            logger.error(f"PMC search failed | error={e}")
            return []
    
    def _fetch_metadata(self, pmc_ids: List[str]) -> List[Dict]:
        """Fetch metadata for list of PMC IDs."""
        if not pmc_ids:
            return []
        
        fetch_params = {
            "db": "pmc",
            "id": ",".join(pmc_ids),
            "retmode": "xml",
        }
        
        if self.api_key:
            fetch_params["api_key"] = self.api_key
        
        fetch_url = f"{PMC_BASE_URL}/efetch.fcgi?{urlencode(fetch_params)}"
        
        try:
            response = self.session.get(fetch_url, timeout=60)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            papers = []
            
            for article in root.findall(".//article"):
                paper = self._parse_article_xml(article)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.error(f"Failed to fetch metadata | error={e}")
            return []
    
    def _parse_article_xml(self, article: ET.Element) -> Optional[Dict]:
        """Parse article XML to extract metadata."""
        try:
            # Get PMC ID
            pmc_id = None
            for article_id in article.findall(".//article-id"):
                if article_id.get("pub-id-type") == "pmc":
                    pmc_id = article_id.text
                    break
            
            if not pmc_id:
                return None
            
            # Get title
            title_elem = article.find(".//article-title")
            title = "".join(title_elem.itertext()) if title_elem is not None else "Unknown"
            
            # Get abstract
            abstract_elem = article.find(".//abstract")
            abstract = "".join(abstract_elem.itertext()) if abstract_elem is not None else ""
            
            # Get publication date
            pub_date = article.find(".//pub-date")
            if pub_date is not None:
                year = pub_date.find("year")
                month = pub_date.find("month")
                day = pub_date.find("day")
                
                year_str = year.text if year is not None else "2020"
                month_str = month.text if month is not None else "01"
                day_str = day.text if day is not None else "01"
                
                date_str = f"{year_str}-{month_str.zfill(2)}-{day_str.zfill(2)}"
            else:
                date_str = "2020-01-01"
            
            # Get figures
            figures = []
            for fig in article.findall(".//fig"):
                fig_data = self._parse_figure(fig, pmc_id)
                if fig_data:
                    figures.append(fig_data)
            
            return {
                "pmc_id": pmc_id,
                "title": title.strip(),
                "abstract": abstract.strip()[:500],
                "published_date": date_str,
                "figures": figures,
                "figure_count": len(figures),
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse article | error={e}")
            return None
    
    def _parse_figure(self, fig: ET.Element, pmc_id: str) -> Optional[Dict]:
        """Parse figure element to extract image info."""
        try:
            fig_id = fig.get("id", "")
            
            # Get caption
            caption_elem = fig.find(".//caption")
            caption = "".join(caption_elem.itertext()) if caption_elem is not None else ""
            
            # Get label (Figure 1, etc.)
            label_elem = fig.find(".//label")
            label = label_elem.text if label_elem is not None else ""
            
            # Get graphic URL
            graphic = fig.find(".//graphic")
            if graphic is not None:
                # xlink:href attribute contains the image path
                href = graphic.get("{http://www.w3.org/1999/xlink}href", "")
                
                if href:
                    return {
                        "fig_id": fig_id,
                        "label": label.strip(),
                        "caption": caption.strip(),
                        "graphic_href": href,
                        "pmc_id": pmc_id,
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to parse figure | error={e}")
            return None
    
    def download_figures(
        self,
        paper: Dict,
        output_dir: Optional[Path] = None,
    ) -> Generator[ChartImage, None, None]:
        """
        Download all figures from a paper.
        
        Args:
            paper: Paper metadata dict from search()
            output_dir: Output directory for images
            
        Yields:
            ChartImage objects
        """
        if output_dir is None:
            output_dir = IMAGES_DIR / "pmc"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_dir = METADATA_DIR / "pmc"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        pmc_id = paper["pmc_id"]
        figures = paper.get("figures", [])
        
        if not figures:
            logger.debug(f"No figures in paper | pmc_id={pmc_id}")
            return
        
        logger.info(
            f"Downloading figures | pmc_id={pmc_id} | count={len(figures)}"
        )
        
        for fig in figures:
            try:
                chart_image = self._download_figure(
                    fig, paper, output_dir, metadata_dir
                )
                if chart_image:
                    yield chart_image
                    
            except Exception as e:
                logger.warning(
                    f"Failed to download figure | fig_id={fig.get('fig_id')} | "
                    f"error={e}"
                )
            
            time.sleep(self.request_delay)
    
    def _download_figure(
        self,
        fig: Dict,
        paper: Dict,
        output_dir: Path,
        metadata_dir: Path,
    ) -> Optional[ChartImage]:
        """Download a single figure."""
        pmc_id = paper["pmc_id"]
        fig_id = fig.get("fig_id", "unknown")
        graphic_href = fig.get("graphic_href", "")
        
        if not graphic_href:
            return None
        
        # Construct image URL
        # PMC images are at: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{id}/bin/{graphic_href}.jpg
        image_url = (
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
            f"bin/{graphic_href}.jpg"
        )
        
        # Try different extensions
        extensions = [".jpg", ".png", ".gif", ".tif"]
        
        for ext in extensions:
            test_url = image_url.replace(".jpg", ext)
            
            try:
                response = self.session.get(test_url, timeout=30)
                
                if response.status_code == 200:
                    # Save image
                    image_id = f"pmc_{pmc_id}_{fig_id}"
                    image_path = output_dir / f"{image_id}{ext}"
                    
                    with open(image_path, "wb") as f:
                        f.write(response.content)
                    
                    # Infer chart type from caption
                    caption = fig.get("caption", "")
                    chart_type = self._infer_chart_type(caption)
                    
                    # Create ChartImage
                    chart_image = ChartImage(
                        image_id=image_id,
                        image_path=image_path,
                        source=DataSource.PMC,
                        source_url=test_url,
                        chart_type=chart_type,
                        caption=caption,
                        status=ProcessingStatus.DOWNLOADED,
                        quality_score=0.85,  # PMC figures are generally high quality
                        metadata={
                            "pmc_id": pmc_id,
                            "paper_title": paper.get("title", ""),
                            "fig_label": fig.get("label", ""),
                            "published_date": paper.get("published_date", ""),
                        },
                    )
                    
                    # Save metadata
                    meta_path = metadata_dir / f"{image_id}.json"
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(
                            chart_image.model_dump(mode="json"),
                            f,
                            indent=2,
                            ensure_ascii=False,
                            default=str,
                        )
                    
                    logger.debug(
                        f"Downloaded | image_id={image_id} | type={chart_type.value}"
                    )
                    
                    return chart_image
                    
            except Exception as e:
                continue
        
        logger.warning(f"Could not download figure | pmc_id={pmc_id} | fig_id={fig_id}")
        return None
    
    def _infer_chart_type(self, caption: str) -> ChartType:
        """Infer chart type from caption text."""
        caption_lower = caption.lower()
        
        type_keywords = {
            ChartType.BAR: ["bar chart", "bar graph", "bar plot", "histogram"],
            ChartType.LINE: ["line chart", "line graph", "line plot", "trend"],
            ChartType.PIE: ["pie chart", "pie graph", "proportion"],
            ChartType.SCATTER: ["scatter plot", "scatter chart", "correlation"],
            ChartType.AREA: ["area chart", "area graph", "stacked area"],
            ChartType.HISTOGRAM: ["histogram", "distribution", "frequency"],
        }
        
        for chart_type, keywords in type_keywords.items():
            if any(kw in caption_lower for kw in keywords):
                return chart_type
        
        # Check for generic visualization words
        if any(word in caption_lower for word in ["graph", "chart", "plot", "figure"]):
            return ChartType.UNKNOWN
        
        return ChartType.UNKNOWN
    
    def hunt(self, limit: int = 100) -> Generator[ChartImage, None, None]:
        """
        Complete hunting workflow: search + download.
        
        Args:
            limit: Total number of images to download
            
        Yields:
            ChartImage objects
        """
        logger.info(f"Starting PMC hunt | target={limit}")
        
        # Search for papers
        papers = self.search(limit=limit * 2)  # Get more papers than needed
        
        if not papers:
            logger.warning("No papers found")
            return
        
        logger.info(f"Found {len(papers)} papers with figures")
        
        # Download figures
        count = 0
        for paper in papers:
            if count >= limit:
                break
            
            for chart_image in self.download_figures(paper):
                count += 1
                yield chart_image
                
                if count >= limit:
                    break
        
        logger.success(f"PMC hunt complete | downloaded={count}")
