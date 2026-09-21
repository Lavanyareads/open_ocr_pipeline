"""
Tests for SpatialAssociator and CrossImageAssociator modules.
"""

import pytest
from app.models.schemas import OCRBlock, DeclarationStatus
from app.extraction.candidates import CandidateExtractor
from app.extraction.spatial_association import SpatialAssociator, CrossImageAssociator


class TestSpatialAssociation:
    """Test suite for spatial proximity and cross-image association."""

    def test_horizontal_spatial_association(self):
        # Anchor block and value block on the same row to the right
        blocks = [
            OCRBlock(text="MRP", bbox=[100, 500, 40, 20], image_index=0),
            OCRBlock(text="₹49.00", bbox=[145, 500, 70, 20], image_index=0),
        ]

        extractor = CandidateExtractor()
        candidates = extractor.extract_all_candidates(blocks)

        # Before spatial association, mrp anchor is UNRESOLVED
        anchor_cand = [c for c in candidates["mrp"] if c.metadata.get("type") == "anchor_only"]
        assert len(anchor_cand) == 1
        assert anchor_cand[0].status == DeclarationStatus.UNRESOLVED

        # Run spatial association
        associator = SpatialAssociator()
        enhanced = associator.associate_candidates(blocks, candidates)

        resolved_mrp = [c for c in enhanced["mrp"] if c.status == DeclarationStatus.RESOLVED]
        assert len(resolved_mrp) > 0
        assert resolved_mrp[0].value == 49.0
        assert resolved_mrp[0].metadata.get("type") == "spatial_association"
        assert len(resolved_mrp[0].source_blocks) == 2

    def test_mrp_separate_number_block_with_vertical_drift(self):
        # Real-world case: "M.R.P. ₹:" at [231, 849, 90, 28] and separate price "373.17" at [498, 824, 112, 24]
        blocks = [
            OCRBlock(text="M.R.P. ₹:", bbox=[231, 849, 90, 28], image_index=0),
            OCRBlock(text="373.17", bbox=[498, 824, 112, 24], image_index=0),
            OCRBlock(text="(Incl. of all taxes)", bbox=[231, 883, 165, 28], image_index=0),
        ]

        extractor = CandidateExtractor()
        candidates = extractor.extract_all_candidates(blocks)

        associator = SpatialAssociator()
        enhanced = associator.associate_candidates(blocks, candidates)

        resolved_mrp = [c for c in enhanced["mrp"] if c.status == DeclarationStatus.RESOLVED]
        assert len(resolved_mrp) > 0
        assert resolved_mrp[0].value == 373.17
        assert resolved_mrp[0].metadata.get("currency") == "INR"

    def test_vertical_spatial_association(self):
        # Anchor block with value directly below
        blocks = [
            OCRBlock(text="Net Quantity", bbox=[100, 500, 100, 20], image_index=0),
            OCRBlock(text="1 kg", bbox=[100, 525, 50, 20], image_index=0),
        ]

        extractor = CandidateExtractor()
        candidates = extractor.extract_all_candidates(blocks)

        associator = SpatialAssociator()
        enhanced = associator.associate_candidates(blocks, candidates)

        resolved_qty = [c for c in enhanced["net_quantity"] if c.status == DeclarationStatus.RESOLVED]
        assert len(resolved_qty) > 0
        assert resolved_qty[0].value == "1 kg"
        assert resolved_qty[0].metadata.get("numeric_value") == 1.0

    def test_cross_image_association(self):
        # MRP label on Image 0, price on Image 1
        blocks = [
            OCRBlock(text="MRP", bbox=[100, 500, 40, 20], image_index=0),
            OCRBlock(text="₹99.00", bbox=[200, 300, 60, 20], image_index=1),
        ]

        extractor = CandidateExtractor()
        candidates = extractor.extract_all_candidates(blocks)

        spatial_associator = SpatialAssociator()
        candidates = spatial_associator.associate_candidates(blocks, candidates)

        cross_associator = CrossImageAssociator()
        cross_enhanced = cross_associator.associate_cross_image(blocks, candidates)

        cross_mrp = [c for c in cross_enhanced["mrp"] if c.metadata.get("type") == "cross_image_association"]
        assert len(cross_mrp) > 0
        assert cross_mrp[0].metadata.get("anchor_image") == 0
        assert cross_mrp[0].metadata.get("value_image") == 1
        # Cross image should remain UNRESOLVED per requirement (not fabricated into resolved)
        assert cross_mrp[0].status == DeclarationStatus.UNRESOLVED
