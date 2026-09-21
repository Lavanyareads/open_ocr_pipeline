"""
Tests for CandidateExtractor module.
"""

import pytest
from app.models.schemas import OCRBlock, DeclarationStatus
from app.extraction.candidates import CandidateExtractor


class TestCandidateExtractor:
    """Test suite for candidate extraction logic across all 11 declarations."""

    @pytest.fixture
    def extractor(self):
        return CandidateExtractor()

    def test_mrp_with_value_in_same_block(self, extractor):
        blocks = [
            OCRBlock(text="MRP ₹49.00", bbox=[100, 500, 110, 20], image_index=0)
        ]
        candidates = extractor.extract_all_candidates(blocks)
        assert len(candidates["mrp"]) > 0
        cand = candidates["mrp"][0]
        assert cand.value == 49.0
        assert cand.status == DeclarationStatus.RESOLVED
        assert cand.metadata.get("currency") == "INR"

    def test_mrp_anchor_only(self, extractor):
        blocks = [
            OCRBlock(text="M.R.P. (incl. of all taxes)", bbox=[100, 500, 150, 20], image_index=0)
        ]
        candidates = extractor.extract_all_candidates(blocks)
        assert len(candidates["mrp"]) > 0
        cand = candidates["mrp"][0]
        assert cand.value is None
        assert cand.status == DeclarationStatus.UNRESOLVED
        assert cand.metadata.get("type") == "anchor_only"

    def test_mrp_value_only(self, extractor):
        blocks = [
            OCRBlock(text="₹199.50", bbox=[100, 500, 80, 20], image_index=0)
        ]
        candidates = extractor.extract_all_candidates(blocks)
        assert len(candidates["mrp"]) > 0
        cand = candidates["mrp"][0]
        assert cand.value == 199.50
        assert cand.status == DeclarationStatus.UNRESOLVED
        assert cand.metadata.get("type") == "value_only"

    def test_net_quantity_anchor_with_value(self, extractor):
        blocks = [
            OCRBlock(text="Net Qty: 500 g", bbox=[100, 600, 120, 20], image_index=0)
        ]
        candidates = extractor.extract_all_candidates(blocks)
        assert len(candidates["net_quantity"]) > 0
        cand = candidates["net_quantity"][0]
        assert cand.value == "500 g"
        assert cand.status == DeclarationStatus.RESOLVED
        assert cand.metadata.get("unit") == "g"
        assert cand.metadata.get("numeric_value") == 500.0

    def test_net_weight_priority_over_serving_size(self, extractor):
        blocks = [
            OCRBlock(text="Serving size: 36 g", bbox=[1, 1, 100, 20], image_index=0),
            OCRBlock(text="Net Wt.: 100 g", bbox=[1, 30, 100, 20], image_index=0),
        ]

        candidates = extractor.extract_all_candidates(blocks)

        assert [candidate.value for candidate in candidates["net_quantity"]] == [
            "100 g"
        ]

    def test_manufacturer_anchor_with_value(self, extractor):
        blocks = [
            OCRBlock(
                text="Manufactured by ABC Foods Pvt Ltd, Plot 42, Mumbai",
                bbox=[100, 700, 300, 20],
                image_index=0,
            )
        ]
        candidates = extractor.extract_all_candidates(blocks)
        assert len(candidates["manufacturer_packer_importer"]) > 0
        cand = candidates["manufacturer_packer_importer"][0]
        assert "ABC Foods" in cand.value
        assert cand.status == DeclarationStatus.RESOLVED

    def test_mkt_by_is_marketer_and_packaging_material_is_excluded(self, extractor):
        blocks = [
            OCRBlock(
                text="Mkt. By: Mondelez India Foods Private Limited",
                bbox=[100, 700, 300, 20], image_index=0,
            ),
            OCRBlock(
                text="Packaging Material Mfd. By: Huhtamaki India Ltd.",
                bbox=[100, 740, 300, 20], image_index=0,
            ),
        ]

        candidates = extractor.extract_all_candidates(blocks)

        assert len(candidates["manufacturer_packer_importer"]) == 1
        assert candidates["manufacturer_packer_importer"][0].value == (
            "Mondelez India Foods Private Limited"
        )

    def test_country_of_origin_made_in(self, extractor):
        blocks = [
            OCRBlock(text="Made in India", bbox=[100, 800, 100, 20], image_index=0)
        ]
        candidates = extractor.extract_all_candidates(blocks)
        assert len(candidates["country_of_origin"]) > 0
        cand = candidates["country_of_origin"][0]
        assert "India" in cand.value
        assert cand.status == DeclarationStatus.RESOLVED
        assert cand.raw_text == "Made in India"
        assert cand.metadata["evidence_type"] == "made_in_statement"

    @pytest.mark.parametrize(
        ("text", "country", "evidence_type"),
        [
            (
                "Country of Origin: India",
                "India",
                "country_of_origin_statement",
            ),
            (
                "Manufactured in India by: Innova Captab Ltd.",
                "India",
                "manufactured_in_statement",
            ),
        ],
    )
    def test_explicit_country_evidence(
        self, extractor, text, country, evidence_type
    ):
        candidates = extractor.extract_all_candidates([
            OCRBlock(text=text, bbox=[100, 800, 300, 20], image_index=0)
        ])

        assert len(candidates["country_of_origin"]) == 1
        candidate = candidates["country_of_origin"][0]
        assert candidate.value == country
        assert candidate.raw_text == text
        assert candidate.metadata["evidence_type"] == evidence_type

    def test_country_of_origin_does_not_infer_from_unrelated_text(self, extractor):
        candidates = extractor.extract_all_candidates([
            OCRBlock(
                text="Manufactured by Innova Captab Ltd.",
                bbox=[100, 800, 300, 20],
                image_index=0,
            )
        ])

        assert candidates["country_of_origin"] == []

    def test_manufactured_country_evidence_includes_split_manufacturer_text(self, extractor):
        blocks = [
            OCRBlock(
                text="Manufactured in India by:",
                bbox=[100, 800, 210, 27],
                image_index=0,
            ),
            OCRBlock(
                text="Innova Captab Ltd.",
                bbox=[100, 827, 151, 25],
                image_index=0,
            ),
        ]

        candidate = extractor.extract_all_candidates(blocks)["country_of_origin"][0]

        assert candidate.value == "India"
        assert candidate.raw_text == (
            "Manufactured in India by: Innova Captab Ltd."
        )
        assert len(candidate.source_blocks) == 2
        assert candidate.metadata["evidence_type"] == "manufactured_in_statement"

    def test_date_candidates(self, extractor):
        blocks = [
            OCRBlock(text="Mfg Date: 15/08/2026", bbox=[100, 300, 180, 20], image_index=0)
        ]
        candidates = extractor.extract_all_candidates(blocks)
        assert len(candidates["manufacture_pack_import_date"]) > 0
        cand = candidates["manufacture_pack_import_date"][0]
        assert cand.value == "15/08/2026"
        assert cand.status == DeclarationStatus.RESOLVED

    def test_best_before_candidates(self, extractor):
        blocks = [
            OCRBlock(text="Best Before 12/2027", bbox=[100, 350, 150, 20], image_index=0)
        ]
        candidates = extractor.extract_all_candidates(blocks)
        assert len(candidates["best_before_use_by"]) > 0
        cand = candidates["best_before_use_by"][0]
        assert cand.value == "12/2027"
        assert cand.status == DeclarationStatus.RESOLVED

    def test_consumer_care_candidates(self, extractor):
        blocks = [
            OCRBlock(
                text="Customer Care: 1800-123-4567, care@company.com",
                bbox=[100, 400, 320, 20],
                image_index=0,
            )
        ]
        candidates = extractor.extract_all_candidates(blocks)
        assert len(candidates["consumer_care"]) > 0
        cand = candidates["consumer_care"][0]
        assert cand.status == DeclarationStatus.RESOLVED
        assert "1800-123-4567" in cand.metadata.get("phones", []) or len(cand.metadata.get("phones", [])) > 0
        assert "care@company.com" in cand.metadata.get("emails", [])

    def test_unit_sale_price_candidates(self, extractor):
        blocks = [
            OCRBlock(text="Unit Sale Price: ₹0.20 / g", bbox=[100, 450, 200, 20], image_index=0)
        ]
        candidates = extractor.extract_all_candidates(blocks)
        assert len(candidates["unit_sale_price"]) > 0
        cand = candidates["unit_sale_price"][0]
        assert cand.status == DeclarationStatus.RESOLVED

    def test_brand_conservative_extraction(self, extractor):
        # Arbitrary product header should NOT be automatically assigned as brand
        blocks = [
            OCRBlock(text="DELICIOUS CHOCO COOKIES", bbox=[100, 50, 300, 40], image_index=0),
            OCRBlock(text="Brand: ChocoDelight", bbox=[100, 100, 150, 20], image_index=0)
        ]
        candidates = extractor.extract_all_candidates(blocks)
        # Should only have brand candidate for the keyword-explicit block
        assert len(candidates["brand_name"]) == 1
        assert "ChocoDelight" in candidates["brand_name"][0].value
