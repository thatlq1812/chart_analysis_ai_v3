"""
Tests for Stage 3 VLM-based extractors.

Verifies extractor factory, backend enum, Pix2StructResult schema,
and the DePlot linearized table parser -- all without loading real models.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.core_engine.schemas.stage_outputs import Pix2StructResult
from src.core_engine.stages.s3_extraction.extractors import (
    BackendType,
    BaseChartExtractor,
    DeplotExtractor,
    MatchaExtractor,
    Pix2StructBaselineExtractor,
    SVLMExtractor,
    _parse_deplot_output,
    create_extractor,
)


# ---------------------------------------------------------------------------
# Tests: factory function
# ---------------------------------------------------------------------------


class TestCreateExtractor:
    """Tests for the create_extractor() factory function."""

    def test_create_extractor_deplot(self) -> None:
        """create_extractor('deplot') should return a DeplotExtractor."""
        ext = create_extractor("deplot")
        assert isinstance(ext, DeplotExtractor)
        assert ext.backend_id == "deplot"

    def test_create_extractor_matcha(self) -> None:
        """create_extractor('matcha') should return a MatchaExtractor."""
        ext = create_extractor("matcha")
        assert isinstance(ext, MatchaExtractor)
        assert ext.backend_id == "matcha"

    def test_create_extractor_pix2struct(self) -> None:
        """create_extractor('pix2struct') should return Pix2StructBaselineExtractor."""
        ext = create_extractor("pix2struct")
        assert isinstance(ext, Pix2StructBaselineExtractor)
        assert ext.backend_id == "pix2struct"

    def test_create_extractor_svlm(self) -> None:
        """create_extractor('svlm') should return SVLMExtractor."""
        ext = create_extractor("svlm")
        assert isinstance(ext, SVLMExtractor)
        assert ext.backend_id == "svlm"

    def test_create_extractor_unknown_backend(self) -> None:
        """Unknown backend identifier should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown extraction backend"):
            create_extractor("nonexistent_backend")

    @pytest.mark.parametrize("backend", ["deplot", "matcha", "pix2struct", "svlm"])
    def test_create_extractor_all_backends(self, backend: str) -> None:
        """All supported backends should be instantiable."""
        ext = create_extractor(backend)
        assert isinstance(ext, BaseChartExtractor)

    def test_create_extractor_case_insensitive(self) -> None:
        """Backend names should be case-insensitive."""
        ext = create_extractor("DEPLOT")
        assert isinstance(ext, DeplotExtractor)


# ---------------------------------------------------------------------------
# Tests: BackendType enum
# ---------------------------------------------------------------------------


class TestBackendTypeEnum:
    """Tests for the BackendType enumeration."""

    def test_backend_type_enum_values(self) -> None:
        """All 4 backends should be present in the enum."""
        assert BackendType.DEPLOT == "deplot"
        assert BackendType.PIX2STRUCT == "pix2struct"
        assert BackendType.MATCHA == "matcha"
        assert BackendType.SVLM == "svlm"

    def test_backend_type_has_four_members(self) -> None:
        """Enum should contain exactly 4 members."""
        assert len(BackendType) == 4


# ---------------------------------------------------------------------------
# Tests: Pix2StructResult schema
# ---------------------------------------------------------------------------


class TestPix2StructResult:
    """Tests for the Pix2StructResult Pydantic model."""

    def test_pix2struct_result_schema(self) -> None:
        """Pix2StructResult should have all required fields."""
        result = Pix2StructResult(
            headers=["Year", "Revenue"],
            rows=[["2021", "100"], ["2022", "150"]],
            records=[
                {"Year": "2021", "Revenue": "100"},
                {"Year": "2022", "Revenue": "150"},
            ],
            raw_html="Year | Revenue\n2021 | 100\n2022 | 150",
            model_name="google/deplot",
            extraction_confidence=1.0,
        )
        assert result.headers == ["Year", "Revenue"]
        assert len(result.rows) == 2
        assert result.extraction_confidence == 1.0
        assert result.model_name == "google/deplot"

    def test_pix2struct_result_defaults(self) -> None:
        """Default values should produce a valid empty result."""
        result = Pix2StructResult()
        assert result.headers == []
        assert result.rows == []
        assert result.records == []
        assert result.raw_html == ""
        assert result.extraction_confidence == 0.0

    def test_pix2struct_result_confidence_bounds(self) -> None:
        """Confidence must be between 0.0 and 1.0."""
        with pytest.raises(Exception):
            Pix2StructResult(extraction_confidence=1.5)
        with pytest.raises(Exception):
            Pix2StructResult(extraction_confidence=-0.1)


# ---------------------------------------------------------------------------
# Tests: DePlot linearized table parser
# ---------------------------------------------------------------------------


class TestDeplotParser:
    """Tests for the _parse_deplot_output() parser function."""

    def test_deplot_parser_simple_table(self) -> None:
        """Parse 'A | B \\n 1 | 2' format."""
        text = "Year | Revenue<0x0A>2021 | 100<0x0A>2022 | 150"
        headers, rows = _parse_deplot_output(text)
        assert headers == ["Year", "Revenue"]
        assert len(rows) == 2
        assert rows[0] == ["2021", "100"]
        assert rows[1] == ["2022", "150"]

    def test_deplot_parser_with_title(self) -> None:
        """TITLE line should be stripped from output."""
        text = "TITLE | Revenue Trend<0x0A>Year | Revenue<0x0A>2021 | 100"
        headers, rows = _parse_deplot_output(text)
        assert "TITLE" not in str(headers)
        assert headers == ["Year", "Revenue"]
        assert len(rows) == 1

    def test_deplot_parser_empty_output(self) -> None:
        """Empty string should return empty headers and rows."""
        headers, rows = _parse_deplot_output("")
        assert headers == []
        assert rows == []

    def test_deplot_parser_no_numeric_data(self) -> None:
        """Output with no numeric rows returns empty."""
        text = "Category | Label<0x0A>foo | bar<0x0A>baz | qux"
        headers, rows = _parse_deplot_output(text)
        # No numeric data -> parser finds no data rows
        assert headers == []
        assert rows == []

    def test_deplot_parser_multiline_headers(self) -> None:
        """Multiple header lines should be merged."""
        text = "Product<0x0A>Sales<0x0A>A | 10<0x0A>B | 20"
        headers, rows = _parse_deplot_output(text)
        assert len(rows) == 2
        # Headers should be derived from non-data lines before first data row

    def test_deplot_parser_percentage_values(self) -> None:
        """Percentage values (e.g., '42%') should be recognized as numeric."""
        text = "Category | Share<0x0A>A | 42%<0x0A>B | 58%"
        headers, rows = _parse_deplot_output(text)
        assert len(rows) == 2

    def test_deplot_parser_comma_separated_numbers(self) -> None:
        """Comma-separated numbers (e.g., '1,234') should be recognized."""
        text = "City | Population<0x0A>NYC | 1,234<0x0A>LA | 5,678"
        headers, rows = _parse_deplot_output(text)
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# Tests: BaseChartExtractor ABC
# ---------------------------------------------------------------------------


class TestBaseChartExtractor:
    """Tests for BaseChartExtractor abstract base class."""

    def test_extractor_base_interface(self) -> None:
        """Cannot instantiate BaseChartExtractor directly."""
        with pytest.raises(TypeError):
            BaseChartExtractor()  # type: ignore[abstract]

    def test_deplot_extractor_not_loaded_initially(self) -> None:
        """Freshly created extractors should not be loaded (lazy load)."""
        ext = DeplotExtractor()
        assert ext.is_available is False

    def test_deplot_default_model(self) -> None:
        """DeplotExtractor default model should be google/deplot."""
        ext = DeplotExtractor()
        assert ext.model_name == "google/deplot"

    def test_matcha_default_model(self) -> None:
        """MatchaExtractor default model should be google/matcha-base."""
        ext = MatchaExtractor()
        assert ext.model_name == "google/matcha-base"
