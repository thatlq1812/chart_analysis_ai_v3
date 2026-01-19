"""
Common Schema Types

Shared types used across all pipeline stages.

NOTE: Enums are defined in enums.py - import from there, not here.
"""

from datetime import datetime
from pathlib import Path
from typing import Tuple

from pydantic import BaseModel, ConfigDict, Field

# Import ChartType from enums.py (Single Source of Truth)
from .enums import ChartType  # noqa: F401 - re-exported for convenience


class BoundingBox(BaseModel):
    """
    Bounding box coordinates in pixel space.
    
    Origin is top-left corner of image.
    """
    
    model_config = ConfigDict(frozen=True)
    
    x_min: int = Field(..., ge=0, description="Left edge pixel coordinate")
    y_min: int = Field(..., ge=0, description="Top edge pixel coordinate")
    x_max: int = Field(..., gt=0, description="Right edge pixel coordinate")
    y_max: int = Field(..., gt=0, description="Bottom edge pixel coordinate")
    confidence: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Detection confidence score",
    )
    
    @property
    def width(self) -> int:
        """Box width in pixels."""
        return self.x_max - self.x_min
    
    @property
    def height(self) -> int:
        """Box height in pixels."""
        return self.y_max - self.y_min
    
    @property
    def area(self) -> int:
        """Box area in pixels squared."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Center point (x, y)."""
        return (
            (self.x_min + self.x_max) // 2,
            (self.y_min + self.y_max) // 2,
        )
    
    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """Return as (x1, y1, x2, y2) tuple."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)
    
    def to_xywh(self) -> Tuple[int, int, int, int]:
        """Return as (x, y, width, height) tuple."""
        return (self.x_min, self.y_min, self.width, self.height)


class Point(BaseModel):
    """2D point in pixel coordinates."""
    
    model_config = ConfigDict(frozen=True)
    
    x: int = Field(..., description="X coordinate")
    y: int = Field(..., description="Y coordinate")
    
    def distance_to(self, other: "Point") -> float:
        """Euclidean distance to another point."""
        import math
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class Color(BaseModel):
    """RGB color representation."""
    
    model_config = ConfigDict(frozen=True)
    
    r: int = Field(..., ge=0, le=255, description="Red channel")
    g: int = Field(..., ge=0, le=255, description="Green channel")
    b: int = Field(..., ge=0, le=255, description="Blue channel")
    
    @property
    def hex(self) -> str:
        """Return hex color string (#RRGGBB)."""
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"
    
    @property
    def rgb_tuple(self) -> Tuple[int, int, int]:
        """Return as (R, G, B) tuple."""
        return (self.r, self.g, self.b)
    
    @classmethod
    def from_hex(cls, hex_str: str) -> "Color":
        """Create Color from hex string."""
        hex_str = hex_str.lstrip("#")
        return cls(
            r=int(hex_str[0:2], 16),
            g=int(hex_str[2:4], 16),
            b=int(hex_str[4:6], 16),
        )
    
    def distance_to(self, other: "Color") -> float:
        """Euclidean distance in RGB space."""
        import math
        return math.sqrt(
            (self.r - other.r) ** 2 +
            (self.g - other.g) ** 2 +
            (self.b - other.b) ** 2
        )


class SessionInfo(BaseModel):
    """
    Processing session metadata.
    
    Created at pipeline start, passed through all stages.
    """
    
    model_config = ConfigDict(frozen=True)
    
    session_id: str = Field(..., description="Unique session identifier")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Session creation timestamp",
    )
    source_file: Path = Field(..., description="Original input file path")
    total_pages: int = Field(default=1, ge=1, description="Total pages in source")
    config_hash: str = Field(
        ...,
        min_length=8,
        max_length=32,
        description="Hash of config for reproducibility",
    )
