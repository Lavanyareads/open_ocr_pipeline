"""
Declaration candidate extraction module.

Identifies candidate text blocks that may correspond to
product declarations (MRP, net quantity, manufacturer, etc.)
using keyword patterns and text analysis.

This module is OCR-engine-independent — it works on normalized
OCR blocks regardless of which engine produced them.

IMPORTANT: This module generates CANDIDATES, not final assignments.
OCR heuristics should never blindly assign semantic meaning.
"""

import re
from typing import List, Dict, Any, Optional

from ..models.schemas import OCRBlock, DeclarationCandidate, DeclarationStatus


# ---------------------------------------------------------------------------
# Anchor patterns — identify declaration LABELS in text
# ---------------------------------------------------------------------------

MRP_PATTERNS = [
    r'\bM\.?R\.?P\.?\b',
    r'\bMaximum\s+Retail\s+Price\b',
    r'\bMax\.?\s+Retail\s+Price\b',
    r'\bRetail\s+Price\b',
]

NET_QUANTITY_PATTERNS = [
    r'\bNet\s+Qu?a?n?t?i?t?y?\b',
    r'\bNet\s+Qty\.?\b',
    r'\bNet\s+Wt\.?\b',
    r'\bNet\s+Weight\b',
    r'\bNet\s+Content[s]?\b',
    r'\bNet\s+Vol(?:ume)?\.?\b',
]

MANUFACTURER_PATTERNS = [
    r'\bManufactured\s+(?:by|&|and)\b',
    r'\bManu?f?a?c?t?u?r?e?d?\s*(?:&|and)?\s*Marketed\s+by\b',
    r'\bPacked\s+by\b',
    r'\bPacked\s*&?\s*(?:Distributed|Marketed)\s+by\b',
    r'\bImported\s+(?:by|&|and)\b',
    r'\bMarketed\s+by\b',
    r'\bMkt\.?\s+By\b',
    r'\bDistributed\s+by\b',
]

COUNTRY_OF_ORIGIN_PATTERNS = [
    r'\bCountry\s+of\s+Origin\b\s*[:.]?\s*([A-Za-z][A-Za-z .-]*)',
    r'\bMade\s+in\b\s+([A-Za-z][A-Za-z .-]*)',
    r'\bManufactured\s+in\b\s+([A-Za-z][A-Za-z .-]*?)(?=\s+by\b|\s*$|:)',
]

DATE_PATTERNS = [
    r'\bMfg\.?\s*(?:Date|Dt\.?)\b',
    r'\bManufactur(?:e|ing)\s*Date\b',
    r'\bDate\s+of\s+Manufactur(?:e|ing)\b',
    r'\bPacking\s*Date\b',
    r'\bPacked?\s+(?:on|Date)\b',
    r'\bDate\s+of\s+Packing\b',
    r'\bImport\s*Date\b',
    r'\bDate\s+of\s+Import\b',
]

BEST_BEFORE_PATTERNS = [
    r'\bBest\s+Before\b',
    r'\bBest\s+By\b',
    r'\bUse\s+Before\b',
    r'\bUse\s+By\b',
    r'\bExp(?:iry)?\.?\s*(?:Date|Dt\.?)\b',
    r'\bExpires?\s+(?:on|by)\b',
    r'\bBB\s*[:.]?\b',
]

CONSUMER_CARE_PATTERNS = [
    r'\bConsumer\s+Care\b',
    r'\bCustomer\s+Care\b',
    r'\bToll\s*[-\s]?\s*Free\b',
    r'\bHelpline\b',
    r'\bGrievance\b',
    r'\bComplaint[s]?\b',
    r'\bContact\s+(?:Us|Details?)\b',
]

UNIT_SALE_PRICE_PATTERNS = [
    r'\bUnit\s+Sale\s+Price\b',
    r'\bPrice\s+per\s+(?:unit|kg|g|ml|l|litre|liter)\b',
]

DIMENSION_PATTERNS = [
    r'\bDimension[s]?\b',
    r'\bLength\s*[x×X]\s*Width\b',
    r'\bL\s*[x×X]\s*W\s*[x×X]\s*H\b',
]

BRAND_PATTERNS = [
    r'\bBrand\s*(?:Name)?\s*:',
    r'\bBrand\s+Name\b',
]

COMMON_NAME_PATTERNS = [
    r'\bCommon\s*Name\b',
    r'\bGeneric\s*Name\b',
    r'\bName\s+of\s+(?:the\s+)?(?:Food|Product|Commodity)\b',
]


# ---------------------------------------------------------------------------
# Value patterns — identify declaration VALUES in text
# ---------------------------------------------------------------------------

PRICE_VALUE_PATTERN = re.compile(
    r'(?:Rs\.?|₹|INR\.?)\s*(\d+(?:[.,]\d+)?)|'
    r'(\d+(?:[.,]\d+)?)\s*(?:Rs\.?|₹|INR)',
    re.IGNORECASE,
)

QUANTITY_VALUE_PATTERN = re.compile(
    r'(\d+(?:[.,]\d+)?)\s*'
    r'(g|gm|gms|gram[s]?|kg|kilogram[s]?|'
    r'ml|mL|millilitre[s]?|milliliter[s]?|'
    r'l|L|litre[s]?|liter[s]?|'
    r'cm|mm|m|inch[es]?|ft|feet|'
    r'pcs?|pieces?|units?|nos?|'
    r'capsule[s]?|tablet[s]?|sachet[s]?|'
    r'wipes?|sheets?|rolls?|packs?|pairs?)',
    re.IGNORECASE,
)

DATE_VALUE_PATTERN = re.compile(
    r'\b(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{2,4})\b|'
    r'\b(\d{1,2})\s*[/.\-]\s*(\d{4})\b|'
    r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*'
    r'[\s,\-]*(\d{1,2})?\s*[\s,\-]*(\d{2,4})\b',
    re.IGNORECASE,
)

PHONE_PATTERN = re.compile(
    r'\b(?:\+91[\s\-]?)?(?:\d[\s\-]?){10,12}\b|'
    r'\b1800[\s\-]?\d{3}[\s\-]?\d{3,4}\b|'
    r'\b\d{3,5}[\s\-]\d{6,8}\b',
)

EMAIL_PATTERN = re.compile(
    r'\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b',
)


class CandidateExtractor:
    """Extracts declaration candidates from normalized OCR blocks.

    Generates candidates for 11 declaration fields by matching
    keyword anchors and value patterns in the OCR text.
    """

    DECLARATION_FIELDS = [
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

    def extract_all_candidates(
        self, blocks: List[OCRBlock]
    ) -> Dict[str, List[DeclarationCandidate]]:
        """Extract declaration candidates from OCR blocks.

        Returns:
            Dict mapping field names to lists of candidates.
        """
        candidates: Dict[str, List[DeclarationCandidate]] = {
            field: [] for field in self.DECLARATION_FIELDS
        }

        candidates["mrp"] = self._extract_mrp_candidates(blocks)
        candidates["net_quantity"] = self._extract_net_quantity_candidates(blocks)
        candidates["manufacturer_packer_importer"] = (
            self._extract_manufacturer_candidates(blocks)
        )
        candidates["country_of_origin"] = (
            self._extract_country_of_origin_candidates(blocks)
        )
        candidates["manufacture_pack_import_date"] = (
            self._extract_date_candidates(blocks)
        )
        candidates["best_before_use_by"] = (
            self._extract_best_before_candidates(blocks)
        )
        candidates["consumer_care"] = self._extract_consumer_care_candidates(blocks)
        candidates["unit_sale_price"] = (
            self._extract_unit_sale_price_candidates(blocks)
        )
        candidates["dimensions"] = self._extract_dimension_candidates(blocks)
        candidates["brand_name"] = self._extract_brand_candidates(blocks)
        candidates["common_name"] = self._extract_common_name_candidates(blocks)

        return candidates

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _match_patterns(text: str, patterns: List[str]) -> Optional[re.Match]:
        """Check if text matches any of the given patterns."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match
        return None

    # ------------------------------------------------------------------
    # Per-field extraction
    # ------------------------------------------------------------------

    def _extract_mrp_candidates(
        self, blocks: List[OCRBlock]
    ) -> List[DeclarationCandidate]:
        candidates: List[DeclarationCandidate] = []
        for block in blocks:
            text = block.text.strip()
            anchor = self._match_patterns(text, MRP_PATTERNS)

            if anchor:
                # Check for price value in same block (with or without currency symbol)
                price = PRICE_VALUE_PATTERN.search(text)
                value = None
                if price:
                    val_str = price.group(1) or price.group(2)
                    try:
                        value = float(val_str.replace(",", ""))
                    except (ValueError, AttributeError):
                        value = None
                else:
                    # Check for trailing numbers after anchor in same block e.g. "MRP: 49.00"
                    after_anchor = text[anchor.end():].strip().lstrip(":").strip()
                    num_match = re.search(r'(\d+(?:\.\d{1,2})?)', after_anchor)
                    if num_match:
                        try:
                            value = float(num_match.group(1).replace(",", ""))
                        except (ValueError, AttributeError):
                            value = None

                if value is not None:
                    candidates.append(DeclarationCandidate(
                        field_name="mrp",
                        value=value,
                        raw_text=text,
                        confidence_score=0.8,
                        source_blocks=[block],
                        status=DeclarationStatus.RESOLVED,
                        metadata={"currency": "INR", "type": "anchor_with_value"},
                    ))
                else:
                    candidates.append(DeclarationCandidate(
                        field_name="mrp",
                        value=None,
                        raw_text=text,
                        confidence_score=0.5,
                        source_blocks=[block],
                        status=DeclarationStatus.UNRESOLVED,
                        metadata={"type": "anchor_only"},
                    ))
            elif PRICE_VALUE_PATTERN.search(text):
                price = PRICE_VALUE_PATTERN.search(text)
                val_str = price.group(1) or price.group(2)
                try:
                    value = float(val_str.replace(",", ""))
                except (ValueError, AttributeError):
                    value = None
                candidates.append(DeclarationCandidate(
                    field_name="mrp",
                    value=value,
                    raw_text=text,
                    confidence_score=0.3,
                    source_blocks=[block],
                    status=DeclarationStatus.UNRESOLVED,
                    metadata={"currency": "INR", "type": "value_only"},
                ))
        return candidates

    def _extract_net_quantity_candidates(
        self, blocks: List[OCRBlock]
    ) -> List[DeclarationCandidate]:
        candidates: List[DeclarationCandidate] = []
        for block in blocks:
            text = block.text.strip()
            # Ignore nutrition / ingredient lines
            if re.search(r'(?i)(?:serving\s+size|no\.\s+of\s+servings|per\s+100|%\s*rda|approx|protein|fat|carb|sugar|sodium|kcal|contains\s*:)', text):
                continue

            anchor = self._match_patterns(text, NET_QUANTITY_PATTERNS)
            if anchor:
                qty = QUANTITY_VALUE_PATTERN.search(text)
                if qty:
                    candidates.append(DeclarationCandidate(
                        field_name="net_quantity",
                        value=f"{qty.group(1)} {qty.group(2)}",
                        raw_text=text,
                        confidence_score=0.8,
                        source_blocks=[block],
                        status=DeclarationStatus.RESOLVED,
                        metadata={
                            "numeric_value": float(qty.group(1).replace(",", "")),
                            "unit": qty.group(2),
                            "type": "anchor_with_value",
                            "priority": 100,
                        },
                    ))
                else:
                    candidates.append(DeclarationCandidate(
                        field_name="net_quantity",
                        value=None,
                        raw_text=text,
                        confidence_score=0.5,
                        source_blocks=[block],
                        status=DeclarationStatus.UNRESOLVED,
                        metadata={"type": "anchor_only"},
                    ))
            elif QUANTITY_VALUE_PATTERN.search(text):
                qty = QUANTITY_VALUE_PATTERN.search(text)
                if qty:
                    candidates.append(DeclarationCandidate(
                        field_name="net_quantity",
                        value=f"{qty.group(1)} {qty.group(2)}",
                        raw_text=text,
                        confidence_score=0.7,
                        source_blocks=[block],
                        status=DeclarationStatus.RESOLVED,
                        metadata={
                            "numeric_value": float(qty.group(1).replace(",", "")),
                            "unit": qty.group(2),
                            "type": "value_only",
                            "priority": 10,
                        },
                    ))
        return candidates

    def _extract_manufacturer_candidates(
        self, blocks: List[OCRBlock]
    ) -> List[DeclarationCandidate]:
        candidates: List[DeclarationCandidate] = []
        for block in blocks:
            text = block.text.strip()
            if re.search(r"\bpackaging\s+material\b", text, re.IGNORECASE):
                continue
            anchor = self._match_patterns(text, MANUFACTURER_PATTERNS)
            if anchor:
                after = text[anchor.end():].strip().lstrip(":").strip()
                candidates.append(DeclarationCandidate(
                    field_name="manufacturer_packer_importer",
                    value=after if after else None,
                    raw_text=text,
                    confidence_score=0.7 if after else 0.5,
                    source_blocks=[block],
                    status=(
                        DeclarationStatus.RESOLVED
                        if after
                        else DeclarationStatus.UNRESOLVED
                    ),
                    metadata={
                        "type": "anchor_with_value" if after else "anchor_only",
                        "anchor_type": anchor.group().strip(),
                    },
                ))
        return candidates

    def _extract_country_of_origin_candidates(
        self, blocks: List[OCRBlock]
    ) -> List[DeclarationCandidate]:
        candidates: List[DeclarationCandidate] = []
        for block in blocks:
            text = block.text.strip()
            match = None
            evidence_type = None
            for pattern, candidate_type in zip(
                COUNTRY_OF_ORIGIN_PATTERNS,
                (
                    "country_of_origin_statement",
                    "made_in_statement",
                    "manufactured_in_statement",
                ),
            ):
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    evidence_type = candidate_type
                    break
            if match and match.group(1):
                country = match.group(1).strip(" .:-")
                source_blocks = [block]
                raw_text = text
                if evidence_type == "manufactured_in_statement":
                    continuation = self._find_country_continuation(blocks, block)
                    if continuation is not None:
                        source_blocks.append(continuation)
                        raw_text = f"{text} {continuation.text.strip()}"
                candidates.append(DeclarationCandidate(
                    field_name="country_of_origin",
                    value=country,
                    raw_text=raw_text,
                    confidence_score=0.8,
                    source_blocks=source_blocks,
                    status=DeclarationStatus.RESOLVED,
                    metadata={
                        "type": evidence_type,
                        "evidence_type": evidence_type,
                    },
                ))
        return candidates

    @staticmethod
    def _find_country_continuation(
        blocks: List[OCRBlock], anchor_block: OCRBlock
    ) -> Optional[OCRBlock]:
        """Find a nearby manufacturer value for complete country evidence text."""
        ax, ay, aw, ah = anchor_block.bbox
        for block in blocks:
            if block is anchor_block or block.image_index != anchor_block.image_index:
                continue
            bx, by, bw, bh = block.bbox
            vertical_gap = by - (ay + ah)
            horizontal_gap = bx - (ax + aw)
            if -40 <= vertical_gap <= 90 and -250 <= horizontal_gap <= 350:
                return block
            if by > ay + ah + 90:
                break
        return None

    def _extract_date_candidates(
        self, blocks: List[OCRBlock]
    ) -> List[DeclarationCandidate]:
        candidates: List[DeclarationCandidate] = []
        for block in blocks:
            text = block.text.strip()
            anchor = self._match_patterns(text, DATE_PATTERNS)
            if anchor:
                date_match = DATE_VALUE_PATTERN.search(text[anchor.end():])
                if date_match:
                    candidates.append(DeclarationCandidate(
                        field_name="manufacture_pack_import_date",
                        value=date_match.group().strip(),
                        raw_text=text,
                        confidence_score=0.8,
                        source_blocks=[block],
                        status=DeclarationStatus.RESOLVED,
                        metadata={
                            "type": "anchor_with_date",
                            "anchor": anchor.group().strip(),
                        },
                    ))
                else:
                    candidates.append(DeclarationCandidate(
                        field_name="manufacture_pack_import_date",
                        value=None,
                        raw_text=text,
                        confidence_score=0.5,
                        source_blocks=[block],
                        status=DeclarationStatus.UNRESOLVED,
                        metadata={
                            "type": "anchor_only",
                            "anchor": anchor.group().strip(),
                        },
                    ))
        return candidates

    def _extract_best_before_candidates(
        self, blocks: List[OCRBlock]
    ) -> List[DeclarationCandidate]:
        candidates: List[DeclarationCandidate] = []
        for block in blocks:
            text = block.text.strip()
            anchor = self._match_patterns(text, BEST_BEFORE_PATTERNS)
            if anchor:
                date_match = DATE_VALUE_PATTERN.search(text[anchor.end():])
                if date_match:
                    candidates.append(DeclarationCandidate(
                        field_name="best_before_use_by",
                        value=date_match.group().strip(),
                        raw_text=text,
                        confidence_score=0.8,
                        source_blocks=[block],
                        status=DeclarationStatus.RESOLVED,
                        metadata={"type": "anchor_with_date"},
                    ))
                else:
                    candidates.append(DeclarationCandidate(
                        field_name="best_before_use_by",
                        value=None,
                        raw_text=text,
                        confidence_score=0.5,
                        source_blocks=[block],
                        status=DeclarationStatus.UNRESOLVED,
                        metadata={"type": "anchor_only"},
                    ))
        return candidates

    def _extract_consumer_care_candidates(
        self, blocks: List[OCRBlock]
    ) -> List[DeclarationCandidate]:
        candidates: List[DeclarationCandidate] = []
        for block in blocks:
            text = block.text.strip()
            anchor = self._match_patterns(text, CONSUMER_CARE_PATTERNS)
            if anchor:
                phones = PHONE_PATTERN.findall(text)
                emails = EMAIL_PATTERN.findall(text)
                after = text[anchor.end():].strip().lstrip(":").strip()
                has_contact = bool(phones or emails)
                candidates.append(DeclarationCandidate(
                    field_name="consumer_care",
                    value=after if (after or has_contact) else None,
                    raw_text=text,
                    confidence_score=0.7 if has_contact else 0.5,
                    source_blocks=[block],
                    status=(
                        DeclarationStatus.RESOLVED
                        if has_contact
                        else DeclarationStatus.UNRESOLVED
                    ),
                    metadata={
                        "type": (
                            "anchor_with_contact"
                            if has_contact
                            else "anchor_only"
                        ),
                        "phones": phones,
                        "emails": emails,
                    },
                ))
            else:
                phones = PHONE_PATTERN.findall(text)
                emails = EMAIL_PATTERN.findall(text)
                if phones or emails:
                    candidates.append(DeclarationCandidate(
                        field_name="consumer_care",
                        value=text,
                        raw_text=text,
                        confidence_score=0.3,
                        source_blocks=[block],
                        status=DeclarationStatus.UNRESOLVED,
                        metadata={
                            "type": "contact_info_only",
                            "phones": phones,
                            "emails": emails,
                        },
                    ))
        return candidates

    def _extract_unit_sale_price_candidates(
        self, blocks: List[OCRBlock]
    ) -> List[DeclarationCandidate]:
        candidates: List[DeclarationCandidate] = []
        for block in blocks:
            text = block.text.strip()
            anchor = self._match_patterns(text, UNIT_SALE_PRICE_PATTERNS)
            if anchor:
                price = PRICE_VALUE_PATTERN.search(text)
                candidates.append(DeclarationCandidate(
                    field_name="unit_sale_price",
                    value=price.group() if price else None,
                    raw_text=text,
                    confidence_score=0.7 if price else 0.5,
                    source_blocks=[block],
                    status=(
                        DeclarationStatus.RESOLVED
                        if price
                        else DeclarationStatus.UNRESOLVED
                    ),
                    metadata={
                        "type": (
                            "anchor_with_value" if price else "anchor_only"
                        ),
                    },
                ))
        return candidates

    def _extract_dimension_candidates(
        self, blocks: List[OCRBlock]
    ) -> List[DeclarationCandidate]:
        candidates: List[DeclarationCandidate] = []
        for block in blocks:
            text = block.text.strip()
            anchor = self._match_patterns(text, DIMENSION_PATTERNS)
            if anchor:
                after = text[anchor.end():].strip()
                candidates.append(DeclarationCandidate(
                    field_name="dimensions",
                    value=after if after else None,
                    raw_text=text,
                    confidence_score=0.5,
                    source_blocks=[block],
                    status=DeclarationStatus.UNRESOLVED,
                    metadata={"type": "anchor_found"},
                ))
        return candidates

    def _extract_brand_candidates(
        self, blocks: List[OCRBlock]
    ) -> List[DeclarationCandidate]:
        """Conservative brand extraction — keyword-triggered only."""
        candidates: List[DeclarationCandidate] = []
        for block in blocks:
            text = block.text.strip()
            match = self._match_patterns(text, BRAND_PATTERNS)
            if match:
                after = text[match.end():].strip().lstrip(":").strip()
                candidates.append(DeclarationCandidate(
                    field_name="brand_name",
                    value=after if after else None,
                    raw_text=text,
                    confidence_score=0.6 if after else 0.4,
                    source_blocks=[block],
                    status=(
                        DeclarationStatus.RESOLVED
                        if after
                        else DeclarationStatus.UNRESOLVED
                    ),
                    metadata={"type": "keyword_match"},
                ))
        return candidates

    def _extract_common_name_candidates(
        self, blocks: List[OCRBlock]
    ) -> List[DeclarationCandidate]:
        """Conservative common-name extraction — keyword-triggered only."""
        candidates: List[DeclarationCandidate] = []
        for block in blocks:
            text = block.text.strip()
            match = self._match_patterns(text, COMMON_NAME_PATTERNS)
            if match:
                after = text[match.end():].strip().lstrip(":").strip()
                candidates.append(DeclarationCandidate(
                    field_name="common_name",
                    value=after if after else None,
                    raw_text=text,
                    confidence_score=0.6 if after else 0.4,
                    source_blocks=[block],
                    status=(
                        DeclarationStatus.RESOLVED
                        if after
                        else DeclarationStatus.UNRESOLVED
                    ),
                    metadata={"type": "keyword_match"},
                ))
        return candidates
