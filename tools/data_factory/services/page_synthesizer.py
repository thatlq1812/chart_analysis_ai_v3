"""
Page Synthesizer - Generate synthetic document pages for YOLO training

This service creates training data by:
1. Extracting text-only pages from PDFs as backgrounds
2. Pasting chart images at random positions
3. Generating accurate YOLO bounding box labels
4. Including NEGATIVE SAMPLES (pages without charts)

Key Features:
- Grayscale output for efficiency (color provides no value for detection)
- Negative samples to prevent false positives
- Optimized for training on 6GB VRAM GPUs
"""

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger
from PIL import Image


@dataclass
class PlacedChart:
    """Information about a chart placed on a page."""

    chart_path: Path
    x: int
    y: int
    width: int
    height: int

    def to_yolo_format(self, page_width: int, page_height: int) -> str:
        """Convert to YOLO format: class x_center y_center width height (normalized)."""
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
    min_chart_width_ratio: float = 0.30
    max_chart_width_ratio: float = 0.85

    # Margins (pixels)
    margin_top: int = 60
    margin_bottom: int = 60
    margin_left: int = 50
    margin_right: int = 50

    # Overlap prevention
    min_gap_between_charts: int = 15
    max_placement_attempts: int = 100

    # Page settings
    page_width: int = 1654  # A4 at 150 DPI
    page_height: int = 2339

    # CRITICAL: Negative sample ratio (pages with NO charts)
    # This teaches YOLO to NOT detect charts on text-only pages
    negative_sample_ratio: float = 0.20  # 20% of dataset are negative samples

    # Output settings
    use_grayscale: bool = True  # Grayscale is more efficient, color has no value
    jpeg_quality: int = 90

    # Augmentation
    apply_noise: bool = True
    noise_intensity: float = 0.015
    random_brightness: bool = True
    brightness_range: Tuple[float, float] = (0.85, 1.15)


class PageSynthesizer:
    """
    Generate synthetic document pages with charts for YOLO training.

    CRITICAL IMPROVEMENTS over previous version:
    1. Grayscale output - color has no detection value
    2. Negative samples - pages WITHOUT charts to prevent false positives
    3. Better bbox labels - accurate positions, not full-page boxes

    Example:
        synthesizer = PageSynthesizer(
            chart_dir=Path("data/academic_dataset/images"),
            background_dir=Path("data/synthetic_source/backgrounds"),
            output_dir=Path("data/training"),
        )
        stats = synthesizer.generate_dataset(num_samples=10000)
    """

    def __init__(
        self,
        chart_dir: Path,
        output_dir: Path,
        background_dir: Optional[Path] = None,
        config: Optional[SynthesizerConfig] = None,
    ):
        self.chart_dir = Path(chart_dir)
        self.output_dir = Path(output_dir)
        self.background_dir = Path(background_dir) if background_dir else None
        self.config = config or SynthesizerConfig()

        # Collect source images
        self.chart_paths = self._collect_images(self.chart_dir)
        if not self.chart_paths:
            raise ValueError(f"No chart images found in {self.chart_dir}")
        logger.info(f"Loaded chart sources | count={len(self.chart_paths)}")

        self.background_paths: List[Path] = []
        if self.background_dir and self.background_dir.exists():
            self.background_paths = self._collect_images(self.background_dir)
            logger.info(f"Loaded background sources | count={len(self.background_paths)}")

        # Setup output directories
        self._setup_output_dirs()

    def _setup_output_dirs(self) -> None:
        """Create output directory structure for YOLO training."""
        for split in ["train", "val"]:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _collect_images(self, directory: Path) -> List[Path]:
        """Collect all image files from directory recursively."""
        extensions = {".png", ".jpg", ".jpeg", ".webp"}
        images = []
        for ext in extensions:
            images.extend(directory.glob(f"**/*{ext}"))
            images.extend(directory.glob(f"**/*{ext.upper()}"))
        return sorted(set(images))

    def _load_background(self) -> Image.Image:
        """Load random background or create blank white page."""
        if self.background_paths:
            bg_path = random.choice(self.background_paths)
            try:
                bg = Image.open(bg_path)
                # Convert to grayscale if needed
                if self.config.use_grayscale:
                    bg = bg.convert("L")
                else:
                    bg = bg.convert("RGB")
                # Resize to target dimensions
                bg = bg.resize(
                    (self.config.page_width, self.config.page_height),
                    Image.Resampling.LANCZOS,
                )
                return bg
            except Exception as e:
                logger.warning(f"Failed to load background | path={bg_path} | error={e}")

        # Fallback to blank page
        mode = "L" if self.config.use_grayscale else "RGB"
        color = 255 if self.config.use_grayscale else (255, 255, 255)
        return Image.new(mode, (self.config.page_width, self.config.page_height), color)

    def _load_chart(self, chart_path: Path) -> Optional[Image.Image]:
        """Load and convert chart image."""
        try:
            chart = Image.open(chart_path)
            if self.config.use_grayscale:
                chart = chart.convert("L")
            else:
                chart = chart.convert("RGB")
            return chart
        except Exception as e:
            logger.warning(f"Failed to load chart | path={chart_path} | error={e}")
            return None

    def _calculate_chart_size(self, chart: Image.Image) -> Tuple[int, int]:
        """Calculate new size maintaining aspect ratio."""
        orig_w, orig_h = chart.size
        aspect_ratio = orig_h / orig_w

        min_w = int(self.config.page_width * self.config.min_chart_width_ratio)
        max_w = int(self.config.page_width * self.config.max_chart_width_ratio)
        new_w = random.randint(min_w, max_w)
        new_h = int(new_w * aspect_ratio)

        # Ensure fits within page
        max_h = self.config.page_height - self.config.margin_top - self.config.margin_bottom
        if new_h > max_h:
            new_h = max_h
            new_w = int(new_h / aspect_ratio)

        return max(new_w, 50), max(new_h, 50)

    def _find_valid_position(
        self,
        chart_w: int,
        chart_h: int,
        placed_charts: List[PlacedChart],
    ) -> Optional[Tuple[int, int]]:
        """Find non-overlapping position for chart."""
        min_x = self.config.margin_left
        max_x = self.config.page_width - self.config.margin_right - chart_w
        min_y = self.config.margin_top
        max_y = self.config.page_height - self.config.margin_bottom - chart_h

        if max_x < min_x or max_y < min_y:
            return None

        for _ in range(self.config.max_placement_attempts):
            x = random.randint(min_x, max_x)
            y = random.randint(min_y, max_y)

            valid = True
            gap = self.config.min_gap_between_charts
            for placed in placed_charts:
                if not (
                    x + chart_w + gap < placed.x
                    or x > placed.x + placed.width + gap
                    or y + chart_h + gap < placed.y
                    or y > placed.y + placed.height + gap
                ):
                    valid = False
                    break

            if valid:
                return x, y

        return None

    def _apply_augmentation(self, image: Image.Image) -> Image.Image:
        """Apply augmentation to final image."""
        img_array = np.array(image, dtype=np.float32)

        # Random brightness
        if self.config.random_brightness:
            factor = random.uniform(*self.config.brightness_range)
            img_array = img_array * factor

        # Random noise
        if self.config.apply_noise:
            noise = np.random.normal(0, self.config.noise_intensity * 255, img_array.shape)
            img_array = img_array + noise

        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def generate_positive_sample(self, sample_id: str, split: str) -> Tuple[Path, Path, int]:
        """Generate a page WITH charts (positive sample)."""
        page = self._load_background()
        num_charts = random.randint(
            self.config.min_charts_per_page,
            self.config.max_charts_per_page,
        )

        placed_charts: List[PlacedChart] = []

        for _ in range(num_charts):
            chart_path = random.choice(self.chart_paths)
            chart = self._load_chart(chart_path)
            if chart is None:
                continue

            new_w, new_h = self._calculate_chart_size(chart)
            chart = chart.resize((new_w, new_h), Image.Resampling.LANCZOS)

            position = self._find_valid_position(new_w, new_h, placed_charts)
            if position is None:
                continue

            x, y = position
            page.paste(chart, (x, y))

            placed_charts.append(
                PlacedChart(
                    chart_path=chart_path,
                    x=x,
                    y=y,
                    width=new_w,
                    height=new_h,
                )
            )

        # Apply augmentation
        page = self._apply_augmentation(page)

        # Save
        image_path = self.output_dir / "images" / split / f"{sample_id}.jpg"
        label_path = self.output_dir / "labels" / split / f"{sample_id}.txt"

        # Convert grayscale to RGB for JPEG (YOLO expects 3 channels)
        if self.config.use_grayscale:
            page = page.convert("RGB")
        page.save(image_path, "JPEG", quality=self.config.jpeg_quality)

        with open(label_path, "w") as f:
            for placed in placed_charts:
                f.write(
                    placed.to_yolo_format(self.config.page_width, self.config.page_height)
                    + "\n"
                )

        return image_path, label_path, len(placed_charts)

    def generate_negative_sample(self, sample_id: str, split: str) -> Tuple[Path, Path]:
        """
        Generate a page WITHOUT any charts (negative sample).

        CRITICAL: This teaches YOLO to NOT detect charts on text-only pages,
        preventing false positives on blank/text regions.
        """
        page = self._load_background()
        page = self._apply_augmentation(page)

        image_path = self.output_dir / "images" / split / f"{sample_id}.jpg"
        label_path = self.output_dir / "labels" / split / f"{sample_id}.txt"

        if self.config.use_grayscale:
            page = page.convert("RGB")
        page.save(image_path, "JPEG", quality=self.config.jpeg_quality)

        # Empty label file (no objects)
        label_path.write_text("")

        return image_path, label_path

    def generate_dataset(
        self,
        num_samples: int,
        train_ratio: float = 0.85,
        prefix: str = "page",
    ) -> dict:
        """
        Generate complete dataset with positive and negative samples.

        Args:
            num_samples: Total number of samples to generate
            train_ratio: Fraction for training set (rest is validation)
            prefix: Filename prefix

        Returns:
            Statistics dictionary
        """
        logger.info("=" * 60)
        logger.info("Starting Synthetic Dataset Generation")
        logger.info("=" * 60)
        logger.info(f"Total samples: {num_samples}")
        logger.info(f"Negative ratio: {self.config.negative_sample_ratio:.0%}")
        logger.info(f"Grayscale: {self.config.use_grayscale}")
        logger.info(f"Train/Val ratio: {train_ratio:.0%}/{1-train_ratio:.0%}")

        num_negative = int(num_samples * self.config.negative_sample_ratio)
        num_positive = num_samples - num_negative

        num_train = int(num_samples * train_ratio)
        num_val = num_samples - num_train

        stats = {
            "total_samples": num_samples,
            "positive_samples": num_positive,
            "negative_samples": num_negative,
            "train_samples": num_train,
            "val_samples": num_val,
            "total_charts_placed": 0,
            "config": {
                "grayscale": self.config.use_grayscale,
                "negative_ratio": self.config.negative_sample_ratio,
                "page_size": f"{self.config.page_width}x{self.config.page_height}",
            },
        }

        # Generate indices
        all_indices = list(range(num_samples))
        random.shuffle(all_indices)

        negative_indices = set(random.sample(all_indices, num_negative))
        train_indices = set(all_indices[:num_train])

        logger.info(f"Generating {num_positive} positive samples...")
        logger.info(f"Generating {num_negative} negative samples...")

        for i, idx in enumerate(all_indices):
            sample_id = f"{prefix}_{idx:06d}"
            split = "train" if idx in train_indices else "val"
            is_negative = idx in negative_indices

            if is_negative:
                self.generate_negative_sample(sample_id, split)
            else:
                _, _, num_charts = self.generate_positive_sample(sample_id, split)
                stats["total_charts_placed"] += num_charts

            if (i + 1) % 500 == 0:
                logger.info(f"Progress: {i + 1}/{num_samples} ({(i + 1) / num_samples:.1%})")

        logger.info("=" * 60)
        logger.info("Dataset generation complete!")
        logger.info(f"Positive samples: {num_positive}")
        logger.info(f"Negative samples: {num_negative}")
        logger.info(f"Total charts placed: {stats['total_charts_placed']}")
        logger.info("=" * 60)

        return stats


def create_dataset_yaml(output_dir: Path, num_classes: int = 1) -> Path:
    """Create YOLO dataset configuration file."""
    yaml_content = f"""# Synthetic Chart Detection Dataset
# Generated by Page Synthesizer

path: {output_dir.resolve()}
train: images/train
val: images/val

nc: {num_classes}
names:
  0: chart
"""
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    logger.info(f"Created dataset.yaml | path={yaml_path}")
    return yaml_path
