"""
Spatial association module.

Uses OCR bounding boxes to associate declaration labels/anchors
with their corresponding values based on spatial proximity.

Includes:
1. SpatialAssociator: Intra-image horizontal/vertical proximity association
2. CrossImageAssociator: Semantic association across multiple images
"""

import re
from typing import List, Dict, Any, Optional, Tuple

from ..models.schemas import (
    OCRBlock,
    DeclarationCandidate,
    DeclarationStatus,
)
from .candidates import (
    PRICE_VALUE_PATTERN,
    QUANTITY_VALUE_PATTERN,
    DATE_VALUE_PATTERN,
    PHONE_PATTERN,
    EMAIL_PATTERN,
)


class SpatialAssociator:
    """Associates declaration anchors with values using spatial proximity on the same image."""

    # Maximum pixel distances for association
    HORIZONTAL_MAX_DISTANCE = 350
    VERTICAL_MAX_DISTANCE = 60
    VERTICAL_BELOW_MAX_DISTANCE = 90

    def associate_candidates(
        self,
        blocks: List[OCRBlock],
        candidates: Dict[str, List[DeclarationCandidate]],
    ) -> Dict[str, List[DeclarationCandidate]]:
        """Enhance candidates by spatially associating anchors with nearby values.

        For unresolved anchor-only candidates, search nearby blocks for compatible values.
        """
        enhanced: Dict[str, List[DeclarationCandidate]] = {}

        for field_name, field_candidates in candidates.items():
            enhanced[field_name] = []
            for candidate in field_candidates:
                if (
                    candidate.status == DeclarationStatus.UNRESOLVED
                    and candidate.metadata.get("type") == "anchor_only"
                ):
                    resolved = self._try_spatial_resolve(
                        field_name, candidate, blocks
                    )
                    enhanced[field_name].append(resolved or candidate)
                else:
                    enhanced[field_name].append(candidate)

        return enhanced

    def _try_spatial_resolve(
        self,
        field_name: str,
        anchor_candidate: DeclarationCandidate,
        blocks: List[OCRBlock],
    ) -> Optional[DeclarationCandidate]:
        """Try to resolve an anchor-only candidate using spatial proximity."""
        if not anchor_candidate.source_blocks:
            return None

        anchor_block = anchor_candidate.source_blocks[0]
        anchor_bbox = anchor_block.bbox
        anchor_image = anchor_block.image_index

        # Find nearby blocks on the same image
        nearby = self._find_nearby_blocks(
            anchor_bbox, anchor_image, blocks, exclude_text=anchor_block.text
        )

        # Try to find a compatible value
        best_match = self._find_best_value_match(field_name, nearby)

        if best_match:
            block, score = best_match
            extracted_val, extra_meta = self._extract_value_for_field(field_name, block.text)
            metadata = {
                **anchor_candidate.metadata,
                "type": "spatial_association",
                "spatial_score": score,
                "associated_text": block.text,
                **extra_meta,
            }
            return DeclarationCandidate(
                field_name=field_name,
                value=extracted_val,
                raw_text=f"{anchor_candidate.raw_text} | {block.text}",
                confidence_score=min(anchor_candidate.confidence_score + 0.3, 0.9),
                source_blocks=anchor_candidate.source_blocks + [block],
                status=DeclarationStatus.RESOLVED if extracted_val is not None else DeclarationStatus.UNRESOLVED,
                metadata=metadata,
            )

        return None

    def _find_nearby_blocks(
        self,
        anchor_bbox: List[float],
        anchor_image: int,
        blocks: List[OCRBlock],
        exclude_text: str = "",
    ) -> List[Tuple[OCRBlock, float]]:
        """Find blocks near the anchor, scored by proximity and line alignment."""
        ax, ay, aw, ah = anchor_bbox
        anchor_center_y = ay + ah / 2.0
        anchor_right = ax + aw
        anchor_bottom = ay + ah

        nearby: List[Tuple[OCRBlock, float]] = []

        for block in blocks:
            if block.text.strip() == exclude_text.strip():
                continue
            if block.image_index != anchor_image:
                continue

            bx, by, bw, bh = block.bbox
            block_center_y = by + bh / 2.0

            # Vertical distance between centers (accounts for line tilt / small differences)
            v_center_diff = abs(anchor_center_y - block_center_y)
            v_overlap = self._vertical_overlap(ay, ah, by, bh)

            # Check horizontal alignment (block to the right of anchor on same line)
            h_distance = bx - anchor_right

            max_row_v_tol = max(ah, bh, 35) * 1.5
            is_same_row = (v_overlap > 0.1) or (v_center_diff <= max_row_v_tol)

            if is_same_row and -30 <= h_distance <= self.HORIZONTAL_MAX_DISTANCE:
                # Closer horizontally and vertically on same row gives highest score
                h_penalty = (max(0.0, h_distance) / self.HORIZONTAL_MAX_DISTANCE) * 0.35
                v_penalty = (v_center_diff / max_row_v_tol) * 0.25
                score = 1.0 - h_penalty - v_penalty
                nearby.append((block, score))
                continue

            # Below the anchor
            v_distance = by - anchor_bottom
            h_overlap = self._horizontal_overlap(ax, aw, bx, bw)
            h_center_diff = abs((ax + aw / 2.0) - (bx + bw / 2.0))

            is_aligned_column = (h_overlap > 0.1) or (h_center_diff <= max(aw, bw, 120))

            if 0 <= v_distance <= self.VERTICAL_BELOW_MAX_DISTANCE and is_aligned_column:
                score = 0.7 - (v_distance / self.VERTICAL_BELOW_MAX_DISTANCE) * 0.35
                nearby.append((block, score))

        nearby.sort(key=lambda x: x[1], reverse=True)
        return nearby

    def _vertical_overlap(
        self, y1: float, h1: float, y2: float, h2: float
    ) -> float:
        """Calculate vertical overlap ratio between two boxes."""
        top = max(y1, y2)
        bottom = min(y1 + h1, y2 + h2)
        overlap = max(0.0, bottom - top)
        min_height = min(h1, h2)
        return overlap / min_height if min_height > 0 else 0.0

    def _horizontal_overlap(
        self, x1: float, w1: float, x2: float, w2: float
    ) -> float:
        """Calculate horizontal overlap ratio between two boxes."""
        left = max(x1, x2)
        right = min(x1 + w1, x2 + w2)
        overlap = max(0.0, right - left)
        min_width = min(w1, w2)
        return overlap / min_width if min_width > 0 else 0.0

    def _find_best_value_match(
        self,
        field_name: str,
        nearby_blocks: List[Tuple[OCRBlock, float]],
    ) -> Optional[Tuple[OCRBlock, float]]:
        """Find the best value match for a field from nearby blocks."""
        def is_valid_price(text: str) -> bool:
            clean = text.strip()
            # If explicit currency symbol
            if PRICE_VALUE_PATTERN.search(clean):
                return True
            # If pure digits / decimals (excluding dates e.g. 05/2026 and barcodes e.g. 8904179737618)
            if not DATE_VALUE_PATTERN.search(clean) and not re.search(r'^\d{8,}$', clean):
                return bool(re.search(r'^\s*(\d+(?:\.\d{1,2})?)\s*$', clean))
            return False

        validators = {
            "mrp": is_valid_price,
            "net_quantity": lambda t: bool(QUANTITY_VALUE_PATTERN.search(t)),
            "manufacture_pack_import_date": lambda t: bool(DATE_VALUE_PATTERN.search(t)),
            "best_before_use_by": lambda t: bool(DATE_VALUE_PATTERN.search(t)),
            "consumer_care": lambda t: bool(PHONE_PATTERN.search(t) or EMAIL_PATTERN.search(t)),
            "unit_sale_price": is_valid_price,
        }

        validator = validators.get(field_name)

        for block, score in nearby_blocks:
            if validator:
                if validator(block.text):
                    return (block, score)
            else:
                if block.text.strip():
                    return (block, score)

        return None

    def _extract_value_for_field(self, field_name: str, text: str) -> Tuple[Any, Dict[str, Any]]:
        """Extract structured value from text for a specific field."""
        extra_meta: Dict[str, Any] = {}
        clean = text.strip()

        if field_name in ("mrp", "unit_sale_price"):
            match = PRICE_VALUE_PATTERN.search(clean)
            if match:
                val_str = match.group(1) or match.group(2)
                try:
                    price_val = float(val_str.replace(",", ""))
                    extra_meta["currency"] = "INR"
                    return price_val, extra_meta
                except (ValueError, AttributeError):
                    pass

            # Standalone numeric price
            num_match = re.search(r'(\d+(?:\.\d{1,2})?)', clean)
            if num_match and not DATE_VALUE_PATTERN.search(clean):
                try:
                    price_val = float(num_match.group(1).replace(",", ""))
                    extra_meta["currency"] = "INR"
                    return price_val, extra_meta
                except (ValueError, AttributeError):
                    pass
            return clean, extra_meta

        elif field_name == "net_quantity":
            match = QUANTITY_VALUE_PATTERN.search(clean)
            if match:
                extra_meta["numeric_value"] = float(match.group(1).replace(",", ""))
                extra_meta["unit"] = match.group(2)
                return f"{match.group(1)} {match.group(2)}", extra_meta

        elif field_name in ("manufacture_pack_import_date", "best_before_use_by"):
            match = DATE_VALUE_PATTERN.search(clean)
            if match:
                return match.group().strip(), extra_meta

        elif field_name == "consumer_care":
            phones = PHONE_PATTERN.findall(clean)
            emails = EMAIL_PATTERN.findall(clean)
            extra_meta["phones"] = phones
            extra_meta["emails"] = emails
            return clean, extra_meta

        return clean, extra_meta


class CrossImageAssociator:
    """Associates declarations across multiple images."""

    FIELD_VALUE_VALIDATORS = {
        "mrp": lambda t: bool(re.search(
            r'(?:Rs\.?|₹|INR)\s*\d+|^\s*\d+(?:[.,]\d+)?\s*$', t, re.IGNORECASE
        )),
        "net_quantity": lambda t: bool(re.search(
            r'\d+\s*(?:g|gm|kg|ml|l|pcs?|units?)', t, re.IGNORECASE
        )),
        "manufacture_pack_import_date": lambda t: bool(re.search(
            r'\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}|'
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',
            t, re.IGNORECASE
        )),
        "best_before_use_by": lambda t: bool(re.search(
            r'\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}|'
            r'\d+\s*(?:months?|days?|years?)',
            t, re.IGNORECASE
        )),
        "consumer_care": lambda t: bool(re.search(
            r'\d{10,}|1800|@|\bmail\b', t, re.IGNORECASE
        )),
    }

    def associate_cross_image(
        self,
        blocks: List[OCRBlock],
        candidates: Dict[str, List[DeclarationCandidate]],
    ) -> Dict[str, List[DeclarationCandidate]]:
        """Try to resolve anchor-only candidates using blocks from other images."""
        enhanced: Dict[str, List[DeclarationCandidate]] = {}

        for field_name, field_candidates in candidates.items():
            enhanced[field_name] = []
            for candidate in field_candidates:
                if (
                    candidate.status == DeclarationStatus.UNRESOLVED
                    and candidate.metadata.get("type") == "anchor_only"
                    and candidate.source_blocks
                ):
                    resolved = self._try_cross_image_resolve(
                        field_name, candidate, blocks
                    )
                    enhanced[field_name].append(resolved or candidate)
                else:
                    enhanced[field_name].append(candidate)

        return enhanced

    def _try_cross_image_resolve(
        self,
        field_name: str,
        anchor_candidate: DeclarationCandidate,
        blocks: List[OCRBlock],
    ) -> Optional[DeclarationCandidate]:
        """Try to resolve using blocks from other images."""
        anchor_image = anchor_candidate.source_blocks[0].image_index
        validator = self.FIELD_VALUE_VALIDATORS.get(field_name)

        if not validator:
            return None

        other_image_blocks = [
            b for b in blocks if b.image_index != anchor_image
        ]

        for block in other_image_blocks:
            if validator(block.text):
                return DeclarationCandidate(
                    field_name=field_name,
                    value=block.text.strip(),
                    raw_text=f"{anchor_candidate.raw_text} | {block.text}",
                    confidence_score=0.4,
                    source_blocks=anchor_candidate.source_blocks + [block],
                    status=DeclarationStatus.UNRESOLVED,
                    metadata={
                        **anchor_candidate.metadata,
                        "type": "cross_image_association",
                        "anchor_image": anchor_image,
                        "value_image": block.image_index,
                    },
                )

        return None
