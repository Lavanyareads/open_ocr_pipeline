from app.models.schemas import OCRBlock, DeclarationCandidate
from app.pipeline import Pipeline


def test_structured_font_measurement_only_keeps_declaration_blocks(tmp_path):
    pipeline = Pipeline.__new__(Pipeline)
    measurement = {
        "pixels_per_mm": 10,
        "text_bboxes": [
            {
                "image_index": 0,
                "text": "Product title",
                "bbox": [1, 1, 20, 20],
                "bbox_height_px": 20,
                "estimated_height_mm": 2,
            },
            {
                "image_index": 0,
                "text": "MRP:",
                "bbox": [1, 30, 20, 10],
                "bbox_height_px": 10,
                "estimated_height_mm": 1,
            },
        ],
    }
        
    candidates = {
        "mrp": [DeclarationCandidate(
            field_name="mrp",
            source_blocks=[
                OCRBlock(text="MRP:", bbox=[1, 30, 20, 10], image_index=0)
            ]
        )],
        "brand_name": [DeclarationCandidate(
            field_name="brand_name",
            source_blocks=[
                OCRBlock(text="Product title", bbox=[1, 1, 20, 20], image_index=0)
            ]
        )],
    }

    result = pipeline._filter_font_measurement(measurement, candidates)

    assert [item["text"] for item in result["text_bboxes"]] == ["MRP:"]
    assert result["pixels_per_mm"] == 10


def test_company_party_blocks_are_not_font_measurements():
    pipeline = Pipeline.__new__(Pipeline)
    measurement = {
        "text_bboxes": [
            {"image_index": 0, "text": "Marketed by :", "bbox": [1, 1, 20, 10]},
            {"image_index": 0, "text": "MRP:", "bbox": [1, 30, 20, 10]},
        ]
    }
    candidates = {
        "manufacturer_packer_importer": [
            DeclarationCandidate(
                field_name="manufacturer_packer_importer",
                source_blocks=[OCRBlock(text="Marketed by :", bbox=[1, 1, 20, 10])],
            )
        ],
        "mrp": [
            DeclarationCandidate(
                field_name="mrp",
                source_blocks=[OCRBlock(text="MRP:", bbox=[1, 30, 20, 10])],
            )
        ],
    }

    result = pipeline._filter_font_measurement(measurement, candidates)

    assert [item["text"] for item in result["text_bboxes"]] == ["MRP:"]