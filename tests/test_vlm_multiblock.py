from app.models.schemas import OCRBlock
from app.vlm.semantic_resolver import VLMSemanticResolver


def test_qwen_visibility_accepts_exact_reading_order_block_span():
    blocks = [
        OCRBlock(text="Potassium Citrate,", bbox=[1, 1, 10, 10]),
        OCRBlock(text="Magnesium Citrate and", bbox=[1, 20, 10, 10]),
        OCRBlock(text="Vitamin B, Oral Solution", bbox=[1, 40, 10, 10]),
    ]

    assert VLMSemanticResolver._value_is_visible(
        "Potassium Citrate, Magnesium Citrate and Vitamin B, Oral Solution",
        blocks,
    )


def test_qwen_visibility_accepts_visible_semantic_blocks_without_correction():
    blocks = [
        OCRBlock(text="Potassium Citrate,", bbox=[1, 1, 10, 10]),
        OCRBlock(text="Magnesium Citrate and", bbox=[1, 20, 10, 10]),
        OCRBlock(text="Vitamin B, Oral Solution", bbox=[1, 40, 10, 10]),
    ]

    assert VLMSemanticResolver._value_is_visible(
        "Potassium Citrate, Magnesium Citrate and Vitamin B, Oral Solution",
        blocks,
    )
    assert not VLMSemanticResolver._value_is_visible(
        "Potassium Citrate, Magnesium Citrate and Vitamin B6 Oral Solution",
        blocks,
    )


def test_qwen_reconstructs_original_ocr_wording_after_visual_correction():
    blocks = [
        OCRBlock(text="Potassium Citrate,", bbox=[1, 1, 10, 10]),
        OCRBlock(text="Magnesium Citrate and", bbox=[1, 20, 10, 10]),
        OCRBlock(text="Vitamin B, Oral Solution", bbox=[1, 40, 10, 10]),
    ]

    assert VLMSemanticResolver._reconstruct_ocr_phrase(
        "Potassium Citrate, Magnesium Citrate and Vitamin B6 Oral Solution",
        blocks,
    ) == "Potassium Citrate, Magnesium Citrate and Vitamin B, Oral Solution"


def test_qwen_reconstruction_does_not_include_unrelated_ocr_blocks():
    blocks = [
        OCRBlock(text="2035548", bbox=[1, 1, 10, 10]),
        OCRBlock(text="450 ml", bbox=[1, 20, 10, 10]),
        OCRBlock(text="CORONA", bbox=[1, 40, 10, 10]),
        OCRBlock(text="Potassium Citrate,", bbox=[1, 60, 10, 10]),
        OCRBlock(text="Magnesium Citrate and", bbox=[1, 80, 10, 10]),
        OCRBlock(text="Vitamin B, Oral Solution", bbox=[1, 100, 10, 10]),
    ]

    assert VLMSemanticResolver._reconstruct_ocr_phrase(
        "Potassium Citrate, Magnesium Citrate and Vitamin B6 Oral Solution",
        blocks,
    ) == "Potassium Citrate, Magnesium Citrate and Vitamin B, Oral Solution"


def test_qwen_prompt_describes_multiblock_reading_order():
    prompt = VLMSemanticResolver(api_key="test")._build_prompt(
        [OCRBlock(text="first", bbox=[1, 1, 10, 10])], {}
    )

    assert "reading order" in prompt
    assert "Never truncate, paraphrase, invent, or omit" in prompt


def test_qwen_rejects_unexplained_numeric_date_codes():
    assert not VLMSemanticResolver._is_explicit_date_value("20222257")
    assert VLMSemanticResolver._is_explicit_date_value("29/06/26")
    assert VLMSemanticResolver._is_explicit_date_value("29/06/27")