"""
Tests for OCR.Space API adapter.
"""

import pytest
import os
import json
from unittest.mock import patch, MagicMock

from app.ocr.ocr_space import OCRSpaceAdapter
from app.models.schemas import OCRBlock


class TestOCRSpaceAdapter:
    """Test suite for OCRSpaceAdapter."""

    def test_init_without_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OCR_SPACE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OCR.Space API key not found"):
            OCRSpaceAdapter(api_key=None)

    def test_init_with_api_key(self):
        adapter = OCRSpaceAdapter(api_key="test_api_key_123")
        assert adapter.api_key == "test_api_key_123"
        assert adapter.ENGINE_NAME == "ocr_space"

    def test_process_nonexistent_image(self):
        adapter = OCRSpaceAdapter(api_key="test_key")
        result = adapter.process_image("non_existent_path_xyz.jpg", image_index=0)
        assert result["success"] is False
        assert result["error_type"] == "file_not_found"
        assert "not found" in result["error"].lower()

    def test_process_unsupported_format(self, tmp_path):
        dummy_file = tmp_path / "test.txt"
        dummy_file.write_text("hello")
        adapter = OCRSpaceAdapter(api_key="test_key")
        result = adapter.process_image(str(dummy_file), image_index=0)
        assert result["success"] is False
        assert result["error_type"] == "unsupported_format"

    @patch("requests.post")
    def test_process_image_success(self, mock_post, tmp_path):
        mock_img = tmp_path / "test.jpg"
        mock_img.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "OCRExitCode": 1,
            "IsErroredOnProcessing": False,
            "ParsedResults": [
                {
                    "TextOverlay": {
                        "Lines": [
                            {
                                "LineText": "MRP Rs. 49.00",
                                "Words": [
                                    {"WordText": "MRP", "Left": 10, "Top": 20, "Width": 30, "Height": 15},
                                    {"WordText": "Rs.", "Left": 45, "Top": 20, "Width": 25, "Height": 15},
                                    {"WordText": "49.00", "Left": 75, "Top": 20, "Width": 40, "Height": 15},
                                ]
                            }
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        adapter = OCRSpaceAdapter(api_key="test_key")
        result = adapter.process_image(str(mock_img), image_index=1)

        assert result["success"] is True
        assert result["image_index"] == 1
        assert result["error"] is None
        assert "ParsedResults" in result["raw_response"]

    def test_normalize_blocks_structure(self):
        adapter = OCRSpaceAdapter(api_key="test_key")
        raw_response = {
            "OCRExitCode": 1,
            "IsErroredOnProcessing": False,
            "ParsedResults": [
                {
                    "TextOverlay": {
                        "Lines": [
                            {
                                "LineText": "MRP Rs. 49.00",
                                "Words": [
                                    {"WordText": "MRP", "Left": 10, "Top": 20, "Width": 30, "Height": 15},
                                    {"WordText": "Rs.", "Left": 45, "Top": 20, "Width": 25, "Height": 15},
                                    {"WordText": "49.00", "Left": 75, "Top": 20, "Width": 40, "Height": 15},
                                ]
                            }
                        ]
                    }
                }
            ]
        }

        blocks = adapter.normalize_blocks(raw_response, image_index=2)
        assert len(blocks) == 1
        b = blocks[0]
        assert b["text"] == "MRP Rs. 49.00"
        assert b["bbox"] == [10, 20, 105, 15]  # min_left=10, min_top=20, max_right=115 -> w=105, max_bottom=35 -> h=15
        assert b["confidence"] is None
        assert b["image_index"] == 2
        assert b["source_engine"] == "ocr_space"
        assert len(b["words"]) == 3

    def test_normalize_empty_response(self):
        adapter = OCRSpaceAdapter(api_key="test_key")
        blocks = adapter.normalize_blocks({}, image_index=0)
        assert blocks == []
