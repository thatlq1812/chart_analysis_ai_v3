"""
Page Synthesizer - Generate synthetic document pages with charts

This service creates training data by pasting chart images onto document
backgrounds, generating accurate YOLO bounding box labels automatically.

Technique: Copy-Paste Augmentation / Synthetic Document Generation
"""

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger
from PIL import Image

from ..config import DataFactoryConfig


@dataclass
class PlacedChart:
    """Information about a chart placed on a page."""
    chart_path: Path
    x: int  # Top-left x coordinate
    y: int  # Top-left y coordinate
    width: int
    height: int
    
    def to_yolo_format(self, page_width: int, page_height: int) -> str:
        """
        Convert to YOLO format: class x_center y_center width height (normalized).
        
        Args:
            page_width: Width of the page image
            page_height: Height of the page image
            
        Returns:
            YOLO format string: "0 x_center y_center w h"
        """
        x_center = (self.x + self.width / 2) / page_width
        y_center = (self.y + self.height / 2) / page_height
        w_norm = self.width / page_width
        h_norm = self.height / page_height
        
        return f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


@dataclass
class SynthesizerConfig:
    """Configuration for page synthesis."""
    # Chart placement
    min_charts_per_page: int = 1
    max_charts_per_page: int = 3
    
    # Chart sizing (relative to page width)
    min_chart_width_ratio: float = 0.3  # Min 30% of page width
    max_chart_width_ratio: float = 0.8  # Max 80% of page width
    
    # Margins (pixels)
    margin_top: int = 50
    margin_bottom: int = 50
    margin_left: int = 50
    margin_right: int = 50
    
    # Overlap prevention
    min_gap_between_charts: int = 20
    max_placement_attempts: int = 50
    
    # Output settings
    page_width: int = 1654  # A4 at 150 DPI
    page_height: int = 2339
    background_color: Tuple[int, int, int] = (255, 255, 255)  # White
    
    # Augmentation
    apply_noise: bool = True
    noise_intensity: float = 0.02
    apply_blur: bool = False
    blur_radius: float = 0.5


class PageSynthesizer:
    """
    Generate synthetic document pages with charts for YOLO training.
    
    This creates realistic training data by:
    1. Loading chart images (foreground)
    2. Creating or loading document backgrounds
    3. Placing charts at random positions
    4. Generating accurate YOLO labels
    
    Example:
        synthesizer = PageSynthesizer(chart_dir, output_dir)
        synthesizer.generate_dataset(num_samples=10000)
    """
    
    def __init__(
        self,
        chart_dir: Path,
        output_dir: Path,
        background_dir: Optional[Path] = None,
        config: Optional[SynthesizerConfig] = None,
    ):
        """
        Initialize the synthesizer.
        
        Args:
            chart_dir: Directory containing chart images (foreground)
            output_dir: Directory to save synthetic pages and labels
            background_dir: Optional directory with background images
            config: Synthesis configuration
        """
        self.chart_dir = Path(chart_dir)
        self.output_dir = Path(output_dir)
        self.background_dir = Path(background_dir) if background_dir else None
        self.config = config or SynthesizerConfig()
        
        # Collect chart images
        self.chart_paths = self._collect_images(self.chart_dir)
        logger.info(f"Found {len(self.chart_paths)} chart images | dir={self.chart_dir}")
        
        # Collect background images if provided
        self.background_paths = []
        if self.background_dir and self.background_dir.exists():
            self.background_paths = self._collect_images(self.background_dir)
            logger.info(f"Found {len(self.background_paths)} background images | dir={self.background_dir}")
        
        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
    
    def _collect_images(self, directory: Path) -> List[Path]:
        """Collect all image files from a directory."""
        extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        images = []
        for ext in extensions:
            images.extend(directory.glob(f"**/*{ext}"))
            images.extend(directory.glob(f"**/*{ext.upper()}"))
        return sorted(images)
    
    def _create_blank_background(self) -> Image.Image:
        """Create a blank white background."""
        return Image.new(
            'RGB',
            (self.config.page_width, self.config.page_height),
            self.config.background_color
        )
    
    def _load_background(self) -> Image.Image:
        """Load a random background or create blank one."""
        if self.background_paths:
            bg_path = random.choice(self.background_paths)
            try:
                bg = Image.open(bg_path).convert('RGB')
                # Resize to target size
                bg = bg.resize(
                    (self.config.page_width, self.config.page_height),
                    Image.Resampling.LANCZOS
                )
                return bg
            except Exception as e:
                logger.warning(f"Failed to load background | path={bg_path} | error={e}")
        
        return self._create_blank_background()
    
    def _calculate_chart_size(self, chart: Image.Image) -> Tuple[int, int]:
        """
        Calculate new size for chart maintaining aspect ratio.
        
        Returns:
            Tuple of (new_width, new_height)
        """
        orig_w, orig_h = chart.size
        aspect_ratio = orig_h / orig_w
        
        # Random width within configured range
        min_w = int(self.config.page_width * self.config.min_chart_width_ratio)
        max_w = int(self.config.page_width * self.config.max_chart_width_ratio)
        new_w = random.randint(min_w, max_w)
        new_h = int(new_w * aspect_ratio)
        
        # Ensure height fits within page
        max_h = self.config.page_height - self.config.margin_top - self.config.margin_bottom
        if new_h > max_h:
            new_h = max_h
            new_w = int(new_h / aspect_ratio)
        
        return new_w, new_h
    
    def _find_valid_position(
        self,
        chart_w: int,
        chart_h: int,
        placed_charts: List[PlacedChart],
    ) -> Optional[Tuple[int, int]]:
        """
        Find a valid position for the chart without overlapping.
        
        Args:
            chart_w: Width of chart to place
            chart_h: Height of chart to place
            placed_charts: List of already placed charts
            
        Returns:
            Tuple of (x, y) or None if no valid position found
        """
        # Available area
        min_x = self.config.margin_left
        max_x = self.config.page_width - self.config.margin_right - chart_w
        min_y = self.config.margin_top
        max_y = self.config.page_height - self.config.margin_bottom - chart_h
        
        if max_x < min_x or max_y < min_y:
            return None
        
        for _ in range(self.config.max_placement_attempts):
            x = random.randint(min_x, max_x)
            y = random.randint(min_y, max_y)
            
            # Check overlap with existing charts
            valid = True
            for placed in placed_charts:
                gap = self.config.min_gap_between_charts
                
                # Check if rectangles overlap (with gap)
                if not (x + chart_w + gap < placed.x or
                        x > placed.x + placed.width + gap or
                        y + chart_h + gap < placed.y or
                        y > placed.y + placed.height + gap):
                    valid = False
                    break
            
            if valid:
                return x, y
        
        return None
    
    def _apply_augmentation(self, image: Image.Image) -> Image.Image:
        """Apply optional augmentation to the final image."""
        img_array = np.array(image)
        
        if self.config.apply_noise:
            noise = np.random.normal(0, self.config.noise_intensity * 255, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def generate_sample(self, sample_id: str) -> Tuple[Path, Path, int]:
        """
        Generate a single synthetic page with charts.
        
        Args:
            sample_id: Unique identifier for this sample
            
        Returns:
            Tuple of (image_path, label_path, num_charts)
        """
        # Load background
        page = self._load_background()
        
        # Decide number of charts
        num_charts = random.randint(
            self.config.min_charts_per_page,
            self.config.max_charts_per_page
        )
        
        placed_charts: List[PlacedChart] = []
        
        for _ in range(num_charts):
            # Select random chart
            chart_path = random.choice(self.chart_paths)
            
            try:
                chart = Image.open(chart_path).convert('RGBA')
            except Exception as e:
                logger.warning(f"Failed to load chart | path={chart_path} | error={e}")
                continue
            
            # Calculate size
            new_w, new_h = self._calculate_chart_size(chart)
            chart = chart.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Find position
            position = self._find_valid_position(new_w, new_h, placed_charts)
            if position is None:
                logger.debug(f"Could not find position for chart {len(placed_charts) + 1}")
                continue
            
            x, y = position
            
            # Paste chart onto page
            # Handle transparency if present
            if chart.mode == 'RGBA':
                page.paste(chart, (x, y), chart)
            else:
                page.paste(chart, (x, y))
            
            # Record placement
            placed_charts.append(PlacedChart(
                chart_path=chart_path,
                x=x,
                y=y,
                width=new_w,
                height=new_h
            ))
        
        # Apply augmentation
        if self.config.apply_noise or self.config.apply_blur:
            page = self._apply_augmentation(page)
        
        # Save image
        image_path = self.images_dir / f"{sample_id}.jpg"
        page.save(image_path, "JPEG", quality=95)
        
        # Save labels
        label_path = self.labels_dir / f"{sample_id}.txt"
        with open(label_path, 'w') as f:
            for placed in placed_charts:
                yolo_line = placed.to_yolo_format(
                    self.config.page_width,
                    self.config.page_height
                )
                f.write(yolo_line + '\n')
        
        return image_path, label_path, len(placed_charts)
    
    def generate_dataset(
        self,
        num_samples: int,
        prefix: str = "synth_page",
        start_index: int = 0,
    ) -> dict:
        """
        Generate a complete synthetic dataset.
        
        Args:
            num_samples: Number of pages to generate
            prefix: Prefix for generated file names
            start_index: Starting index for sample IDs
            
        Returns:
            Dictionary with generation statistics
        """
        logger.info(f"Starting dataset generation | samples={num_samples} | output={self.output_dir}")
        
        stats = {
            "total_samples": 0,
            "total_charts": 0,
            "failed_samples": 0,
            "charts_per_page": [],
        }
        
        for i in range(num_samples):
            sample_id = f"{prefix}_{start_index + i:06d}"
            
            try:
                _, _, num_charts = self.generate_sample(sample_id)
                stats["total_samples"] += 1
                stats["total_charts"] += num_charts
                stats["charts_per_page"].append(num_charts)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Progress | generated={i + 1}/{num_samples}")
                    
            except Exception as e:
                logger.error(f"Failed to generate sample | id={sample_id} | error={e}")
                stats["failed_samples"] += 1
        
        # Calculate statistics
        if stats["charts_per_page"]:
            stats["avg_charts_per_page"] = sum(stats["charts_per_page"]) / len(stats["charts_per_page"])
            stats["min_charts_per_page"] = min(stats["charts_per_page"])
            stats["max_charts_per_page"] = max(stats["charts_per_page"])
        
        del stats["charts_per_page"]  # Remove raw data
        
        # Save manifest
        manifest_path = self.output_dir / "manifest.json"
        manifest = {
            "generated_at": datetime.now().isoformat(),
            "config": {
                "min_charts": self.config.min_charts_per_page,
                "max_charts": self.config.max_charts_per_page,
                "page_size": [self.config.page_width, self.config.page_height],
            },
            "source_charts": len(self.chart_paths),
            "source_backgrounds": len(self.background_paths),
            "stats": stats,
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Dataset generation complete | stats={stats}")
        return stats


def create_dataset_yaml(output_dir: Path, train_ratio: float = 0.8) -> Path:
    """
    Create YOLO dataset.yaml file for the synthetic dataset.
    
    Args:
        output_dir: Directory containing images/ and labels/
        train_ratio: Ratio of training data (rest is validation)
        
    Returns:
        Path to created dataset.yaml
    """
    output_dir = Path(output_dir)
    
    # Split images into train/val
    images_dir = output_dir / "images"
    all_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_ratio)
    
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Create train/val directories
    train_img_dir = output_dir / "train" / "images"
    train_lbl_dir = output_dir / "train" / "labels"
    val_img_dir = output_dir / "val" / "images"
    val_lbl_dir = output_dir / "val" / "labels"
    
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Move files (using symlinks to save space)
    labels_dir = output_dir / "labels"
    
    for img_path in train_images:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        (train_img_dir / img_path.name).symlink_to(img_path)
        if lbl_path.exists():
            (train_lbl_dir / lbl_path.name).symlink_to(lbl_path)
    
    for img_path in val_images:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        (val_img_dir / img_path.name).symlink_to(img_path)
        if lbl_path.exists():
            (val_lbl_dir / lbl_path.name).symlink_to(lbl_path)
    
    # Create dataset.yaml
    yaml_content = f"""# Synthetic Chart Detection Dataset
# Generated by PageSynthesizer

path: {output_dir.absolute()}
train: train/images
val: val/images

# Classes
nc: 1
names:
  0: chart

# Dataset info
# Total images: {len(all_images)}
# Train: {len(train_images)}
# Val: {len(val_images)}
"""
    
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    
    logger.info(f"Created dataset.yaml | train={len(train_images)} | val={len(val_images)}")
    return yaml_path
