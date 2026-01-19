"""
Synthetic Chart Generator - Generate training data using matplotlib

Creates diverse chart images with known ground truth labels
for training and testing chart detection/analysis models.
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from ..config import DataFactoryConfig, IMAGES_DIR, METADATA_DIR, ANNOTATIONS_DIR
from ..schemas import ChartImage, ChartType, DataSource


class SyntheticChartGenerator:
    """
    Generate synthetic chart images with ground truth annotations.
    
    Creates diverse charts for training data augmentation.
    """
    
    def __init__(self, config: DataFactoryConfig):
        self.config = config
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Style variations
        self.color_palettes = [
            plt.cm.tab10.colors,
            plt.cm.Set1.colors,
            plt.cm.Set2.colors,
            plt.cm.Pastel1.colors,
            plt.cm.Dark2.colors,
        ]
        
        self.font_sizes = [8, 10, 12, 14, 16]
        self.figure_sizes = [(6, 4), (8, 6), (10, 6), (8, 8), (12, 8)]
        
        # Ensure output directories
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    def generate_dataset(
        self,
        count: int = 100,
        chart_types: Optional[List[ChartType]] = None,
    ) -> List[ChartImage]:
        """
        Generate a dataset of synthetic charts.
        
        Args:
            count: Number of charts to generate
            chart_types: List of chart types to generate (default: all)
            
        Returns:
            List of generated ChartImage objects
        """
        if chart_types is None:
            chart_types = [
                ChartType.BAR,
                ChartType.LINE,
                ChartType.PIE,
                ChartType.SCATTER,
                ChartType.AREA,
            ]
        
        logger.info(f"Generating synthetic charts | count={count} | types={[t.value for t in chart_types]}")
        
        images: List[ChartImage] = []
        charts_per_type = count // len(chart_types)
        
        for chart_type in chart_types:
            for i in range(charts_per_type):
                try:
                    chart_image = self._generate_chart(chart_type, i)
                    if chart_image:
                        images.append(chart_image)
                except Exception as e:
                    logger.warning(f"Failed to generate chart | type={chart_type} | index={i} | error={e}")
        
        logger.info(f"Generation complete | total={len(images)}")
        return images
    
    def _generate_chart(self, chart_type: ChartType, index: int) -> Optional[ChartImage]:
        """Generate a single chart."""
        
        # Random style parameters
        figsize = random.choice(self.figure_sizes)
        colors = random.choice(self.color_palettes)
        fontsize = random.choice(self.font_sizes)
        
        # Generate data
        n_categories = random.randint(3, 10)
        n_series = random.randint(1, 4) if chart_type in [ChartType.BAR, ChartType.LINE] else 1
        
        data = self._generate_random_data(n_categories, n_series)
        labels = self._generate_labels(n_categories)
        series_names = [f"Series {i+1}" for i in range(n_series)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate chart based on type
        title = f"Sample {chart_type.value.title()} Chart {index + 1}"
        
        if chart_type == ChartType.BAR:
            self._create_bar_chart(ax, data, labels, series_names, colors)
        elif chart_type == ChartType.LINE:
            self._create_line_chart(ax, data, labels, series_names, colors)
        elif chart_type == ChartType.PIE:
            self._create_pie_chart(ax, data[0], labels, colors)
        elif chart_type == ChartType.SCATTER:
            self._create_scatter_chart(ax, data, colors)
        elif chart_type == ChartType.AREA:
            self._create_area_chart(ax, data, labels, series_names, colors)
        else:
            plt.close()
            return None
        
        # Add title and styling
        ax.set_title(title, fontsize=fontsize + 2, fontweight="bold")
        
        # Random style variations
        if random.random() > 0.3 and chart_type not in [ChartType.PIE]:
            ax.grid(True, alpha=random.uniform(0.1, 0.5))
        
        plt.tight_layout()
        
        # Save image
        image_id = f"synthetic_{chart_type.value}_{index:04d}"
        image_path = IMAGES_DIR / f"{image_id}.png"
        
        dpi = random.choice([72, 100, 150])
        fig.savefig(image_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
        
        # Get image dimensions
        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size
        
        # Create ChartImage object
        chart_image = ChartImage(
            image_id=image_id,
            source=DataSource.SYNTHETIC,
            image_path=image_path,
            width=width,
            height=height,
            file_size_bytes=image_path.stat().st_size,
            chart_type=chart_type,
            caption_text=title,
            context_text=f"Synthetic {chart_type.value} chart with {n_categories} categories",
        )
        
        # Save metadata
        metadata_path = METADATA_DIR / f"{image_id}.json"
        with open(metadata_path, "w") as f:
            json.dump(chart_image.model_dump(mode="json"), f, indent=2, default=str)
        
        # Save ground truth annotation (for training)
        annotation = {
            "image_id": image_id,
            "chart_type": chart_type.value,
            "data": {
                "labels": labels,
                "values": [d.tolist() if hasattr(d, "tolist") else d for d in data],
                "series_names": series_names,
            },
            "style": {
                "figsize": figsize,
                "dpi": dpi,
                "fontsize": fontsize,
            },
        }
        annotation_path = ANNOTATIONS_DIR / f"{image_id}.json"
        with open(annotation_path, "w") as f:
            json.dump(annotation, f, indent=2)
        
        return chart_image
    
    def _generate_random_data(self, n_categories: int, n_series: int) -> List[np.ndarray]:
        """Generate random data for charts."""
        data = []
        for _ in range(n_series):
            # Different data patterns
            pattern = random.choice(["uniform", "increasing", "decreasing", "wave"])
            
            if pattern == "uniform":
                values = np.random.uniform(10, 100, n_categories)
            elif pattern == "increasing":
                base = np.linspace(20, 80, n_categories)
                values = base + np.random.uniform(-10, 10, n_categories)
            elif pattern == "decreasing":
                base = np.linspace(80, 20, n_categories)
                values = base + np.random.uniform(-10, 10, n_categories)
            elif pattern == "wave":
                x = np.linspace(0, 2 * np.pi, n_categories)
                values = 50 + 30 * np.sin(x) + np.random.uniform(-5, 5, n_categories)
            
            data.append(values)
        
        return data
    
    def _generate_labels(self, n: int) -> List[str]:
        """Generate category labels."""
        label_types = [
            [f"Category {i+1}" for i in range(n)],
            [f"Item {chr(65+i)}" for i in range(min(n, 26))],
            [f"Q{i+1}" for i in range(n)],
            [f"2024-{i+1:02d}" for i in range(min(n, 12))],
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:n],
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][:n],
        ]
        return random.choice(label_types)[:n]
    
    def _create_bar_chart(
        self,
        ax,
        data: List[np.ndarray],
        labels: List[str],
        series_names: List[str],
        colors,
    ) -> None:
        """Create a bar chart."""
        n_series = len(data)
        n_categories = len(labels)
        
        if n_series == 1:
            # Simple bar chart
            bars = ax.bar(labels, data[0], color=colors[:n_categories])
            
            # Optional: add value labels
            if random.random() > 0.5:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f"{height:.0f}",
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               ha="center", va="bottom", fontsize=8)
        else:
            # Grouped bar chart
            x = np.arange(n_categories)
            width = 0.8 / n_series
            
            for i, (values, name) in enumerate(zip(data, series_names)):
                offset = (i - n_series/2 + 0.5) * width
                ax.bar(x + offset, values, width, label=name, color=colors[i % len(colors)])
            
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
        
        ax.set_ylabel("Value")
        ax.set_xlabel("Category")
    
    def _create_line_chart(
        self,
        ax,
        data: List[np.ndarray],
        labels: List[str],
        series_names: List[str],
        colors,
    ) -> None:
        """Create a line chart."""
        x = range(len(labels))
        
        for i, (values, name) in enumerate(zip(data, series_names)):
            marker = random.choice(["o", "s", "^", "D", "v", ""])
            linestyle = random.choice(["-", "--", "-.", ":"])
            
            ax.plot(x, values, marker=marker, linestyle=linestyle,
                   label=name, color=colors[i % len(colors)], linewidth=2)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45 if len(labels) > 6 else 0)
        ax.set_ylabel("Value")
        ax.set_xlabel("Category")
        
        if len(data) > 1:
            ax.legend()
    
    def _create_pie_chart(
        self,
        ax,
        data: np.ndarray,
        labels: List[str],
        colors,
    ) -> None:
        """Create a pie chart."""
        # Ensure positive values
        values = np.abs(data)
        
        # Random style
        explode = None
        if random.random() > 0.5:
            explode = [0.05 if i == np.argmax(values) else 0 for i in range(len(values))]
        
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            colors=colors[:len(values)],
            autopct="%1.1f%%" if random.random() > 0.3 else None,
            explode=explode,
            shadow=random.random() > 0.7,
            startangle=random.randint(0, 360),
        )
        
        ax.axis("equal")
    
    def _create_scatter_chart(
        self,
        ax,
        data: List[np.ndarray],
        colors,
    ) -> None:
        """Create a scatter chart."""
        x = data[0]
        y = data[1] if len(data) > 1 else np.random.uniform(10, 100, len(x))
        
        # Random scatter parameters
        sizes = np.random.uniform(20, 200, len(x))
        alpha = random.uniform(0.5, 0.9)
        
        ax.scatter(x, y, s=sizes, c=colors[0], alpha=alpha, edgecolors="white")
        
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        
        # Optional: add trend line
        if random.random() > 0.5:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r--", alpha=0.7, label="Trend")
    
    def _create_area_chart(
        self,
        ax,
        data: List[np.ndarray],
        labels: List[str],
        series_names: List[str],
        colors,
    ) -> None:
        """Create an area chart."""
        x = range(len(labels))
        
        if len(data) == 1:
            ax.fill_between(x, data[0], alpha=0.5, color=colors[0])
            ax.plot(x, data[0], color=colors[0], linewidth=2)
        else:
            # Stacked area
            ax.stackplot(x, *data, labels=series_names, colors=colors[:len(data)], alpha=0.7)
            ax.legend(loc="upper left")
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45 if len(labels) > 6 else 0)
        ax.set_ylabel("Value")
        ax.set_xlabel("Category")
