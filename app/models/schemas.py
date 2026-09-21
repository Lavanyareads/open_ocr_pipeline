"""Data models and schemas for the OCR.Space extraction pipeline.

Uses Pydantic v2 for validation and serialization.
BBox convention: [x, y, width, height] in pixels, origin = top-left.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class DeclarationStatus(str, Enum):
    """Status of a declaration extraction."""
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    NOT_DETECTED = "not_detected"


class OCRWord(BaseModel):
    """An individual word detected by OCR."""
    text: str
    bbox: List[float] = Field(description="[x, y, width, height] in pixels")


class OCRBlock(BaseModel):
    """A single normalized OCR text block (line-level)."""
    text: str
    bbox: List[float] = Field(description="[x, y, width, height] in pixels")
    confidence: Optional[float] = Field(
        default=None,
        description="OCR confidence 0.0-1.0. null if engine does not provide it."
    )
    image_index: int = 0
    source_engine: str = "ocr_space"
    words: Optional[List[OCRWord]] = None


class DeclarationCandidate(BaseModel):
    """A candidate extraction for a declaration field."""
    field_name: str
    value: Any = None
    raw_text: str = ""
    confidence_score: float = 0.0
    source_blocks: List[OCRBlock] = Field(default_factory=list)
    status: DeclarationStatus = DeclarationStatus.NOT_DETECTED
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DeclarationValue(BaseModel):
    """Final resolved value for a declaration in structured output."""
    value: Any = None
    status: DeclarationStatus = DeclarationStatus.NOT_DETECTED
    raw_text: Optional[str] = None
    source_images: List[int] = Field(default_factory=list)
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProductExtraction(BaseModel):
    """Complete extraction result for a product."""
    product_id: str
    declarations: Dict[str, DeclarationValue] = Field(default_factory=dict)
    font_measurement: Dict[str, Any] = Field(default_factory=dict)

    def to_structured_json(self) -> Dict[str, Any]:
        """Convert to concise structured JSON for Person 5's Rule Engine.
        
        Keeps structured.json lean:
        - resolved: includes value, raw_text, source_images, and field-specific metadata
        - unresolved: includes candidates and raw_text
        - not_detected: only status
        
        Does NOT include internal scores, all OCR blocks, or compliance conclusions.
        """
        result: Dict[str, Any] = {
            "product_id": self.product_id,
            "declarations": {},
            "font_measurement": self.font_measurement,
        }

        for field_name, decl in self.declarations.items():
            entry: Dict[str, Any] = {"status": decl.status.value}

            if decl.status == DeclarationStatus.RESOLVED:
                entry["value"] = decl.value
                if decl.raw_text:
                    entry["raw_text"] = decl.raw_text
                if decl.source_images:
                    entry["source_images"] = decl.source_images
                # Promote field-specific metadata (e.g. currency, unit)
                if decl.metadata:
                    for key, val in decl.metadata.items():
                        if key not in entry:
                            entry[key] = val

            elif decl.status == DeclarationStatus.UNRESOLVED:
                if decl.candidates:
                    entry["candidates"] = decl.candidates
                if decl.raw_text:
                    entry["raw_text"] = decl.raw_text

            result["declarations"][field_name] = entry

        return result
