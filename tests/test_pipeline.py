"""
Integration tests for the full OCR.Space extraction pipeline.
"""

import pytest
import json
import os
from unittest.mock import patch, MagicMock

from app.models.schemas import OCRBlock, DeclarationStatus
from app.extraction.candidates import CandidateExtractor
from app.extraction.spatial_association import SpatialAssociator
from app.extraction.declarations import DeclarationResolver
from app.pipeline import Pipeline


class TestPipelineIntegration:
    """Integration test suite for end-to-end pipeline."""

    @pytest.fixture
    def mock_ocr_response(self):
        return {
            "OCRExitCode": 1,
            "IsErroredOnProcessing": False,
            "ParsedResults": [
                {
                    "TextOverlay": {
                        "Lines": [
                            {
                                "LineText": "Brand Name: Royal Delight",
                                "Words": [{"WordText": "Brand", "Left": 50, "Top": 50, "Width": 40, "Height": 20}]
                            },
                            {
                                "LineText": "MRP: Rs. 149.00",
                                "Words": [{"WordText": "MRP:", "Left": 50, "Top": 100, "Width": 40, "Height": 20}]
                            },
                            {
                                "LineText": "Net Weight: 250 g",
                                "Words": [{"WordText": "Net", "Left": 50, "Top": 150, "Width": 40, "Height": 20}]
                            },
                            {
                                "LineText": "Made in India",
                                "Words": [{"WordText": "Made", "Left": 50, "Top": 200, "Width": 40, "Height": 20}]
                            },
                            {
                                "LineText": "Customer Care: 1800-999-0000",
                                "Words": [{"WordText": "Customer", "Left": 50, "Top": 250, "Width": 60, "Height": 20}]
                            },
                        ]
                    }
                }
            ]
        }

    @patch("app.ocr.ocr_space.OCRSpaceAdapter.process_image")
    def test_full_pipeline_run(self, mock_process, mock_ocr_response, tmp_path):
        # Create a mock image file in input_dir
        input_dir = tmp_path / "input_images"
        input_dir.mkdir()
        img_file = input_dir / "pack_front.jpg"
        img_file.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF")

        output_dir = tmp_path / "output"

        mock_process.return_value = {
            "image_path": str(img_file),
            "image_index": 0,
            "success": True,
            "error": None,
            "error_type": None,
            "raw_response": mock_ocr_response,
        }

        pipeline = Pipeline(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            api_key="mock_key",
        )

        result = pipeline.run(product_id="test_product_101")

        assert "error" not in result
        assert result["product_id"] == "test_product_101"

        # Verify structured.json exists and has correct schema
        assert os.path.exists(result["structured_path"])
        with open(result["structured_path"], "r", encoding="utf-8") as f:
            structured_data = json.load(f)
            assert structured_data["product_id"] == "test_product_101"
            assert "raw_ocr_path" not in result
            assert len(structured_data["ocr_text"]["images"]) == 1
            assert len(structured_data["ocr_text"]["images"][0]["text_blocks"]) == 5
            assert structured_data["ocr_text"]["images"][0]["text_blocks"][0] == {
                "text": "Brand Name: Royal Delight",
                "bbox": [50, 50, 40, 20],
                "confidence": None,
                "image_index": 0,
            }
            assert structured_data["product_classification"] == {
                "category": "unknown",
                "confidence": 0.0,
                "source": "qwen3_vl_semantic_resolver",
            }
            declarations = structured_data["declarations"]

            # Exact declaration field names
            expected_fields = [
                "brand_name",
                "common_name",
                "manufacturer_packer_importer",
                "country_of_origin",
                "net_quantity",
                "mrp",
                "manufacture_pack_import_date",
                "best_before_use_by",
                "consumer_care",
                "unit_sale_price",
                "dimensions",
            ]
            for field in expected_fields:
                assert field in declarations
                assert declarations[field]["status"] in ("resolved", "unresolved", "not_detected")

            # Check resolved fields
            assert declarations["mrp"]["status"] == "resolved"
            assert declarations["mrp"]["value"] == 149.0
            assert declarations["mrp"]["currency"] == "INR"

            assert declarations["net_quantity"]["status"] == "resolved"
            assert declarations["net_quantity"]["value"] == "250 g"

            assert declarations["country_of_origin"]["status"] == "resolved"
            assert "India" in declarations["country_of_origin"]["value"]

            assert declarations["brand_name"]["status"] == "resolved"
            assert "Royal Delight" in declarations["brand_name"]["value"]

            assert declarations["consumer_care"]["status"] == "resolved"

            # Check undetected fields
            assert declarations["dimensions"]["status"] == "not_detected"

    def test_structured_json_clean_separation(self):
        resolver = DeclarationResolver()
        blocks = [
            OCRBlock(text="MRP ₹49.00", bbox=[10, 20, 30, 40], image_index=0)
        ]
        extractor = CandidateExtractor()
        candidates = extractor.extract_all_candidates(blocks)
        product_extraction = resolver.resolve("prod_01", candidates)

        structured = product_extraction.to_structured_json()
        assert "product_id" in structured
        assert "declarations" in structured
        # Must not leak internal engine metadata into structured.json
        assert "source_engine" not in str(structured)
        assert "confidence_score" not in str(structured["declarations"]["mrp"])
