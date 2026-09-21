from app.vlm.semantic_resolver import VLMSemanticResolver


def test_product_classification_accepts_only_fixed_categories():
    resolver = VLMSemanticResolver(api_key="test")

    assert resolver._validate_classification({
        "product_classification": {"category": "food", "confidence": 0.85}
    }) == {"category": "food", "confidence": 0.85}
    assert resolver._validate_classification({
        "product_classification": {"category": "invented", "confidence": 1.5}
    }) == {"category": "unknown", "confidence": 0.0}


def test_product_classification_clamps_confidence():
    resolver = VLMSemanticResolver(api_key="test")

    assert resolver._validate_classification({
        "product_classification": {"category": "food", "confidence": 4}
    }) == {"category": "food", "confidence": 1.0}