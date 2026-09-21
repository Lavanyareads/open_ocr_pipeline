from app.extraction.declarations import DeclarationResolver
from app.models.schemas import (
    DeclarationCandidate,
    DeclarationStatus,
    OCRBlock,
)
from app.pipeline import Pipeline


def _candidate(text, value, anchor, image_index=0):
    block = OCRBlock(text=text, bbox=[10, 20, 100, 20], image_index=image_index)
    return DeclarationCandidate(
        field_name="manufacturer_packer_importer",
        value=value,
        raw_text=text,
        confidence_score=0.8,
        source_blocks=[block],
        status=DeclarationStatus.RESOLVED,
        metadata={"anchor_type": anchor},
    )


def test_manufacturer_roles_are_serialized_as_entries():
    result = DeclarationResolver().resolve(
        "alka juice",
        {
            "manufacturer_packer_importer": [
                _candidate("Marketed by : CORONA Remedies Limited", "CORONA Remedies Limited", "Marketed by"),
                _candidate("Manufactured in India by: Innova Captab Ltd.", "Innova Captab Ltd.", "Manufactured in India by"),
            ]
        },
    ).to_structured_json()

    declaration = result["declarations"]["manufacturer_packer_importer"]
    assert declaration["status"] == "resolved"
    assert declaration["entries"] == [
        {
            "role": "marketer",
            "value": "CORONA Remedies Limited",
            "raw_text": "Marketed by : CORONA Remedies Limited",
            "source_images": [0],
        },
        {
            "role": "manufacturer",
            "value": "Innova Captab Ltd.",
            "raw_text": "Manufactured in India by: Innova Captab Ltd.",
            "source_images": [0],
        },
    ]


def test_ambiguous_mrp_remains_unresolved_with_candidates():
    blocks = [
        OCRBlock(text="MRP Rs. 55", bbox=[1, 1, 10, 10]),
        OCRBlock(text="MRP Rs. 60", bbox=[1, 20, 10, 10]),
    ]
    candidates = {
        "mrp": [
            DeclarationCandidate(
                field_name="mrp", value=55.0, raw_text=blocks[0].text,
                status=DeclarationStatus.RESOLVED, source_blocks=[blocks[0]],
            ),
            DeclarationCandidate(
                field_name="mrp", value=60.0, raw_text=blocks[1].text,
                status=DeclarationStatus.RESOLVED, source_blocks=[blocks[1]],
            ),
        ]
    }

    result = DeclarationResolver().resolve("product", candidates).to_structured_json()

    assert result["declarations"]["mrp"]["status"] == "unresolved"
    assert len(result["declarations"]["mrp"]["candidates"]) == 2


def test_marketer_can_resolve_without_manufacturer():
    candidate = _candidate(
        "Mkt. By: Mondelez India Foods Private Limited",
        "Mondelez India Foods Private Limited",
        "Mkt. By",
    )

    result = DeclarationResolver().resolve(
        "product", {"manufacturer_packer_importer": [candidate]}
    ).to_structured_json()

    assert result["declarations"]["manufacturer_packer_importer"] == {
        "status": "resolved",
        "entries": [{
            "role": "marketer",
            "value": "Mondelez India Foods Private Limited",
            "raw_text": "Mkt. By: Mondelez India Foods Private Limited",
            "source_images": [0],
        }],
    }


def test_product_name_is_qwen_only_and_stays_unresolved_without_qwen():
    result = DeclarationResolver().resolve(
        "product",
        {"product_name": []},
    ).to_structured_json()

    assert result["declarations"]["product_name"] == {
        "status": "not_detected"
    }


def test_additional_product_information_preserves_ocr_evidence():
    blocks = [
        OCRBlock(text="Batch No.:", bbox=[1, 2, 100, 20], image_index=0),
        OCRBlock(text="L1166020", bbox=[120, 22, 80, 20], image_index=0),
        OCRBlock(text="WARNING: Keep away from", bbox=[1, 50, 200, 20], image_index=0),
        OCRBlock(text="children", bbox=[1, 72, 80, 20], image_index=0),
        OCRBlock(text="Pineapple", bbox=[1, 100, 100, 20], image_index=0),
        OCRBlock(text="Flavour", bbox=[105, 100, 80, 20], image_index=0),
    ]

    result = Pipeline._build_additional_product_information(blocks)

    assert [item["role"] for item in result] == [
        "batch_number", "warning", "flavour"
    ]
    assert result[0]["value"] == "L1166020"
    assert result[0]["raw_text"] == "Batch No.: L1166020"
    assert result[1]["value"] == "WARNING: Keep away from children"
    assert result[2]["value"] == "Pineapple Flavour"


def test_additional_product_information_deduplicates_repeated_statements():
    blocks = [
        OCRBlock(text="STORE IN A COOL PLACE", bbox=[1, 1, 100, 20], image_index=0),
        OCRBlock(text="STORE IN A COOL PLACE", bbox=[1, 1, 100, 20], image_index=1),
    ]

    result = Pipeline._build_additional_product_information(blocks)

    assert len(result) == 1
    assert result[0]["role"] == "instruction"
    assert result[0]["source_images"] == [0, 1]