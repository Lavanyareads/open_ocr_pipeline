"""
Main pipeline module.

Orchestrates the full OCR → extraction → structured output pipeline.

Architecture:
    approved images → OCR.Space → normalized OCR blocks
    → candidate extraction → spatial association → structured.json

The pipeline is modular so another OCR engine can be added later
without rewriting extraction logic.
"""

import os
import json
import glob
import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .ocr.ocr_space import OCRSpaceAdapter
from .extraction.candidates import CandidateExtractor
from .extraction.spatial_association import SpatialAssociator, CrossImageAssociator
from .extraction.declarations import DeclarationResolver
from .models.schemas import OCRBlock, DeclarationStatus, EvidenceEntry
from .vlm.semantic_resolver import VLMSemanticResolver
from .measurement.barcode_calibration import BarcodeCalibrator

logger = logging.getLogger(__name__)


class Pipeline:
    """Main OCR extraction pipeline.
    
    Processes product images through:
    1. OCR.Space API → raw text + bounding boxes
    2. Normalization → engine-independent OCR blocks
    3. Candidate extraction → declaration candidates via regex/keywords
    4. Spatial association → label-value pairing using bboxes
    5. Cross-image association → declarations spanning multiple images
    6. Declaration resolution → final structured output
    """

    SUPPORTED_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".bmp",
        ".tiff", ".tif", ".gif", ".webp", ".pdf",
    }
    FONT_MEASUREMENT_FIELDS = {
        "mrp",
        "net_quantity",
        "country_of_origin",
        "manufacture_pack_import_date",
        "best_before_use_by",
        "consumer_care",
        "unit_sale_price",
        "dimensions",
    }

    def __init__(
        self,
        input_dir: str = "input_images",
        output_dir: str = "output",
        api_key: Optional[str] = None,
        barcode_width_mm: Optional[float] = None,
        barcode_height_mm: Optional[float] = None,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.structured_dir = os.path.join(output_dir, "structured_json")

        os.makedirs(self.structured_dir, exist_ok=True)

        self.ocr_adapter = OCRSpaceAdapter(api_key=api_key)
        self.candidate_extractor = CandidateExtractor()
        self.spatial_associator = SpatialAssociator()
        self.cross_image_associator = CrossImageAssociator()
        self.declaration_resolver = DeclarationResolver()
        self.vlm_resolver = VLMSemanticResolver()
        self.barcode_calibrator = BarcodeCalibrator(
            barcode_width_mm=barcode_width_mm,
            barcode_height_mm=barcode_height_mm,
        )

    def run(self, product_id: Optional[str] = None) -> Dict[str, Any]:
        """Run the full pipeline.
        
        Args:
            product_id: Optional product identifier. Auto-generated if not provided.
            
        Returns:
            Dict with pipeline results including paths to output files.
        """
        product_dirs = self._discover_product_dirs()
        if product_id is None and product_dirs:
            results = []
            for product_dir in product_dirs:
                results.append(self._run_product(
                    product_id=os.path.basename(os.path.normpath(product_dir)),
                    images=self._discover_images(product_dir),
                ))
            return {"products": results}

        if not product_id:
            product_id = f"product_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return self._run_product(product_id, self._discover_images())

    def _run_product(self, product_id: str, images: List[str]) -> Dict[str, Any]:
        """Run every stage for one product folder."""

        print(f"\n{'=' * 60}")
        print(f"Pipeline started for product: {product_id}")
        print(f"{'=' * 60}")

        # Step 1: Discover images
        if not images:
            print(f"No images found in '{self.input_dir}'.")
            return {"error": "No images found", "product_id": product_id}

        print(f"\nFound {len(images)} image(s):")
        for i, img in enumerate(images):
            print(f"  [{i}] {os.path.basename(img)}")

        # Step 2: OCR processing
        print(f"\n--- OCR Processing ---")
        all_blocks: List[Dict[str, Any]] = []
        raw_responses: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        for idx, image_path in enumerate(images):
            print(f"\nProcessing image [{idx}]: {os.path.basename(image_path)}")
            result = self.ocr_adapter.process_image(image_path, image_index=idx)
            raw_responses.append(result)

            if result["success"]:
                blocks = self.ocr_adapter.normalize_blocks(
                    result["raw_response"], image_index=idx
                )
                all_blocks.extend(blocks)
                print(f"  -> {len(blocks)} OCR blocks extracted")
            else:
                errors.append({
                    "image_index": idx,
                    "image_path": image_path,
                    "error": result["error"],
                    "error_type": result.get("error_type"),
                })
                print(f"  -> ERROR: {result['error']}")

        # Step 3: Print normalized blocks
        print(f"\n--- Normalized OCR Blocks ({len(all_blocks)} total) ---")
        for block in all_blocks:
            text_preview = block['text'][:60]
            if len(block['text']) > 60:
                text_preview += '...'
            print(
                f"  [img {block['image_index']}] "
                f"text='{text_preview}' "
                f"bbox={block['bbox']} "
                f"confidence={block['confidence']} "
                f"engine={block['source_engine']}"
            )

        # Step 4: Declaration candidate extraction
        print(f"\n--- Declaration Extraction ---")
        ocr_block_models = [OCRBlock(**b) for b in all_blocks]

        candidates = self.candidate_extractor.extract_all_candidates(
            ocr_block_models
        )

        for field, field_candidates in candidates.items():
            if field_candidates:
                print(f"  {field}: {len(field_candidates)} candidate(s)")
                for c in field_candidates:
                    print(
                        f"    -> value={c.value}, status={c.status.value}, "
                        f"conf={c.confidence_score:.2f}, "
                        f"type={c.metadata.get('type')}"
                    )

        # Step 6: Spatial association (intra-image)
        print(f"\n--- Spatial Association ---")
        candidates = self.spatial_associator.associate_candidates(
            ocr_block_models, candidates
        )

        # Step 7: Cross-image association
        print(f"\n--- Cross-Image Association ---")
        candidates = self.cross_image_associator.associate_cross_image(
            ocr_block_models, candidates
        )

        # Step 8: Resolve semantic and ambiguous declarations with Qwen.
        print(f"\n--- VLM Semantic Resolution ---")
        vlm_results = self.vlm_resolver.resolve_semantics(
            images, ocr_block_models, candidates
        )

        # Keep complete measurement evidence in memory, but only expose
        # declaration-related block measurements in structured JSON.
        full_font_measurement = self.barcode_calibrator.measure(
            images, ocr_block_models
        )

        # Step 6: Resolve declarations
        print(f"\n--- Declaration Resolution ---")
        extraction = self.declaration_resolver.resolve(
            product_id, candidates, vlm_results=vlm_results
        )
        extraction.declarations["manufacturer_packer_importer"] = (
            self._augment_party_entries(
                extraction.declarations["manufacturer_packer_importer"],
                ocr_block_models,
            )
        )

        # Physical measurement is independent of declaration resolution.
        extraction.font_measurement = self._filter_font_measurement(
            full_font_measurement, candidates
        )
        extraction.ocr_text = self._build_ocr_text(images, ocr_block_models)
        extraction.additional_product_information = [
            EvidenceEntry(**item)
            for item in self._build_additional_product_information(ocr_block_models)
        ]
        extraction.product_classification = {
            **vlm_results.get(
                "product_classification",
                {"category": "unknown", "confidence": 0.0},
            ),
            "source": "qwen3_vl_semantic_resolver",
        }

        # Step 7: Save the single complete structured artifact
        structured = extraction.to_structured_json()
        structured_path = os.path.join(
            self.structured_dir, f"{product_id}.json"
        )
        with open(structured_path, "w", encoding="utf-8") as f:
            json.dump(structured, f, indent=2, ensure_ascii=False)
        print(f"\nStructured JSON saved to: {structured_path}")

        # Step 8: Print summary
        print(f"\n{'=' * 60}")
        print("EXTRACTION SUMMARY")
        print(f"{'=' * 60}")
        for field, decl in extraction.declarations.items():
            status_icon = {
                "resolved": "✓",
                "unresolved": "?",
                "not_detected": "✗",
            }.get(decl.status.value, " ")
            print(
                f"  [{status_icon}] {field}: "
                f"status={decl.status.value}, "
                f"value={decl.value if not decl.entries else None}"
            )
            for entry in decl.entries:
                print(f"    {entry.role}={entry.value}")
        print(f"{'=' * 60}\n")

        return {
            "product_id": product_id,
            "images_processed": len(images),
            "ocr_blocks": len(all_blocks),
            "structured_path": structured_path,
            "extraction": structured,
        }

    @staticmethod
    def _build_ocr_text(
        images: List[str], blocks: List[OCRBlock]
    ) -> Dict[str, Any]:
        """Group every normalized OCR block by source image."""
        grouped = {
            index: [] for index in range(len(images))
        }
        for block in blocks:
            grouped.setdefault(block.image_index, []).append({
                "text": block.text,
                "bbox": block.bbox,
                "confidence": block.confidence,
                "image_index": block.image_index,
            })
        return {
            "images": [
                {
                    "image_index": image_index,
                    "text_blocks": grouped[image_index],
                }
                for image_index in sorted(grouped)
            ]
        }

    @staticmethod
    def _build_additional_product_information(
        blocks: List[OCRBlock],
    ) -> List[Dict[str, Any]]:
        """Preserve useful non-legal evidence without semantic guessing."""
        patterns = [
            ("batch_number", r"\bbatch\s*(?:no|number)\b\s*[:.]?\s*(.*)"),
            ("manufacturing_license", r"\bmfg\.?\s*lic(?:ense)?\.?\s*no\.?\s*[:.]?\s*(.*)"),
            ("formulation", r"\bformulation\b\s*[:.]?\s*(.*)"),
            ("ingredient", r"\b(?:ingredients?|composition|contains?)\b\s*[:.]?\s*(.*)"),
            ("warning", r"\b(?:warning|caution)\b\s*[:.]?\s*(.*)"),
            ("instruction", r"\b(?:keep|shake|use|store|directions?)\b.*"),
            ("flavour", r"\b(?:flavou?r)\b\s*[:.]?\s*(.*)"),
            ("variant", r"\bvariant\b\s*[:.]?\s*(.*)"),
        ]
        information = []
        for block_index, block in enumerate(blocks):
            text = block.text.strip()
            lower_text = text.casefold()
            role = None
            value = text
            for candidate_role, pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    role = candidate_role
                    value = text if candidate_role == "instruction" else match.group(1).strip(" .:") or text
                    break
            if role == "flavour" and value == text:
                previous_block = Pipeline._find_previous_block(blocks, block_index)
                if previous_block is not None:
                    value = f"{previous_block.text.strip()} {text}"
                    raw_text = value
            if role and (value == text or role in {"warning", "instruction"}):
                next_block = Pipeline._find_adjacent_block(blocks, block_index)
                if next_block is not None:
                    continuation = next_block.text.strip()
                    if value == text and role not in {"warning", "instruction"}:
                        value = continuation
                    elif continuation.casefold() not in value.casefold():
                        value = f"{text if role == 'warning' else value} {continuation}".strip()
                    raw_text = f"{text} {continuation}"
                else:
                    raw_text = text
            else:
                raw_text = text
            if role is None and any(
                phrase in lower_text
                for phrase in ("sugar free", "100% natural", "no added sugar")
            ):
                role = "marketing_claim"
            if role:
                information.append({
                    "role": role,
                    "value": value,
                    "raw_text": raw_text,
                    "source_images": [block.image_index],
                })
        return Pipeline._deduplicate_product_information(information)

    @staticmethod
    def _deduplicate_product_information(
        information: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge repeated or overlapping evidence statements."""
        merged: List[Dict[str, Any]] = []
        for item in information:
            role = item["role"]
            value = str(item["value"]).strip()
            normalized = re.sub(r"\s+", " ", value.casefold()).strip()
            match = next(
                (
                    existing
                    for existing in merged
                    if existing["role"] == role
                    and (
                        normalized == existing["_normalized"]
                        or normalized in existing["_normalized"]
                        or existing["_normalized"] in normalized
                    )
                ),
                None,
            )
            if match is None:
                item = dict(item)
                item["_normalized"] = normalized
                merged.append(item)
                continue
            if len(normalized) > len(match["_normalized"]):
                match.update({
                    "value": item["value"],
                    "raw_text": item["raw_text"],
                    "_normalized": normalized,
                })
            match["source_images"] = sorted(set(
                match["source_images"] + item["source_images"]
            ))
        for item in merged:
            item.pop("_normalized", None)
        return merged

    @staticmethod
    def _find_previous_block(
        blocks: List[OCRBlock], block_index: int
    ) -> Optional[OCRBlock]:
        """Find a preceding OCR line that continues a short label phrase."""
        block = blocks[block_index]
        bx, by, bw, bh = block.bbox
        for previous in reversed(blocks[:block_index]):
            if previous.image_index != block.image_index:
                continue
            px, py, pw, ph = previous.bbox
            vertical_gap = by - (py + ph)
            horizontal_gap = bx - (px + pw)
            if vertical_gap <= 90 and -250 <= horizontal_gap <= 350:
                return previous
            if py < by - 90:
                break
        return None

    @staticmethod
    def _find_adjacent_block(
        blocks: List[OCRBlock], block_index: int
    ) -> Optional[OCRBlock]:
        """Find the next nearby OCR line for a label-only evidence block."""
        block = blocks[block_index]
        bx, by, bw, bh = block.bbox
        for following in blocks[block_index + 1:]:
            if following.image_index != block.image_index:
                continue
            fx, fy, fw, fh = following.bbox
            vertical_gap = fy - (by + bh)
            horizontal_gap = fx - (bx + bw)
            if -40 <= vertical_gap <= 90 and horizontal_gap <= 350:
                return following
            if fy > by + bh + 90:
                break
        return None

    @staticmethod
    def _augment_party_entries(
        declaration: Any, blocks: List[OCRBlock]
    ) -> Any:
        """Add role entries from label/value OCR lines only at output time."""
        role_patterns = [
            ("manufacturer", r"\bmanufactured\b"),
            ("packer", r"\bpacked\b"),
            ("importer", r"\bimported\b"),
            ("marketer", r"\bmarketed\b"),
            ("marketer", r"\bmkt\.?\s+by\b"),
            ("distributor", r"\bdistributed\b"),
        ]
        entries = list(declaration.entries)
        existing = {(entry.role, entry.value) for entry in entries}
        for index, block in enumerate(blocks):
            if re.search(r"\bpackaging\s+material\b", block.text, re.IGNORECASE):
                continue
            role = next(
                (role for role, pattern in role_patterns
                 if re.search(pattern, block.text, re.IGNORECASE)),
                None,
            )
            if role is None:
                continue
            label_match = re.search(
                r"(?:manufactured|packed|imported|marketed|mkt\.?\s+by|distributed)"
                r"\s*(?:in\s+[A-Za-z]+\s+)?(?:by)?\s*[:.]?\s*(.*)$",
                block.text,
                re.IGNORECASE,
            )
            if label_match and label_match.group(1).strip():
                continue
            next_block = Pipeline._find_adjacent_block(blocks, index)
            if next_block is None:
                continue
            value = next_block.text.strip()
            if (role, value) in existing:
                for entry in entries:
                    if entry.role == role and entry.value == value:
                        entry.raw_text = f"{block.text.strip()} {value}"
                continue
            entries.append(EvidenceEntry(
                role=role,
                value=value,
                raw_text=f"{block.text.strip()} {value}",
                source_images=[block.image_index],
            ))
            existing.add((role, value))
        if entries:
            declaration.status = DeclarationStatus.RESOLVED
            declaration.entries = entries
        return declaration

    def _filter_font_measurement(
        self,
        measurement: Dict[str, Any],
        candidates: Dict[str, List[Any]],
    ) -> Dict[str, Any]:
        """Retain only bbox measurements tied to legal declaration candidates."""
        relevant_keys = {
            (
                block.image_index,
                block.text,
                tuple(block.bbox),
            )
            for field, field_candidates in candidates.items()
            if field in self.FONT_MEASUREMENT_FIELDS
            for candidate in field_candidates
            for block in candidate.source_blocks
        }

        filtered = {
            key: value
            for key, value in measurement.items()
            if key != "text_bboxes"
        }
        filtered["text_bboxes"] = [
            bbox
            for bbox in measurement.get("text_bboxes", [])
            if (
                bbox.get("image_index"),
                bbox.get("text"),
                tuple(bbox.get("bbox", [])),
            ) in relevant_keys
        ]
        return filtered

    def _discover_images(self, directory: Optional[str] = None) -> List[str]:
        """Discover supported image files directly in one product directory."""
        directory = directory or self.input_dir
        if not os.path.isdir(directory):
            return []

        images: List[str] = []
        for ext in self.SUPPORTED_EXTENSIONS:
            images.extend(glob.glob(os.path.join(directory, f"*{ext}")))
            images.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))

        return sorted(set(images))

    def _discover_product_dirs(self) -> List[str]:
        """Find product folders when input_dir is the product collection root."""
        if not os.path.isdir(self.input_dir):
            return []
        if self._discover_images():
            return []
        return sorted(
            path for path in glob.glob(os.path.join(self.input_dir, "*"))
            if os.path.isdir(path) and self._discover_images(path)
        )
