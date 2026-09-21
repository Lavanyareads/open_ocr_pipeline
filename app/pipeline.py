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
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .ocr.ocr_space import OCRSpaceAdapter
from .extraction.candidates import CandidateExtractor
from .extraction.spatial_association import SpatialAssociator, CrossImageAssociator
from .extraction.declarations import DeclarationResolver
from .models.schemas import OCRBlock
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
        "manufacturer_packer_importer",
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
        self.raw_ocr_dir = os.path.join(output_dir, "raw_ocr")
        self.structured_dir = os.path.join(output_dir, "structured_json")

        os.makedirs(self.raw_ocr_dir, exist_ok=True)
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

        # Step 4: Save raw OCR output
        raw_ocr_output = {
            "product_id": product_id,
            "timestamp": datetime.now().isoformat(),
            "images_processed": len(images),
            "total_blocks": len(all_blocks),
            "raw_responses": [
                {
                    "image_index": r["image_index"],
                    "image_path": r["image_path"],
                    "success": r["success"],
                    "error": r["error"],
                    "raw_response": r["raw_response"],
                }
                for r in raw_responses
            ],
            "normalized_blocks": all_blocks,
            "errors": errors,
        }

        # Step 5: Declaration candidate extraction
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
        raw_ocr_output["vlm_results"] = vlm_results

        # Keep the complete measurement evidence in raw OCR output, but only
        # expose declaration-related block measurements in structured JSON.
        full_font_measurement = self.barcode_calibrator.measure(
            images, ocr_block_models
        )
        raw_ocr_output["font_measurement"] = full_font_measurement

        raw_ocr_path = os.path.join(
            self.raw_ocr_dir, f"{product_id}_raw_ocr.json"
        )
        with open(raw_ocr_path, "w", encoding="utf-8") as f:
            json.dump(raw_ocr_output, f, indent=2, ensure_ascii=False)
        print(f"\nRaw OCR saved to: {raw_ocr_path}")

        # Step 9: Resolve declarations
        print(f"\n--- Declaration Resolution ---")
        extraction = self.declaration_resolver.resolve(
            product_id, candidates, vlm_results=vlm_results
        )

        # Physical measurement is independent of declaration resolution.
        extraction.font_measurement = self._filter_font_measurement(
            full_font_measurement, candidates
        )

        # Step 10: Save structured JSON
        structured = extraction.to_structured_json()
        structured_path = os.path.join(
            self.structured_dir, f"{product_id}.json"
        )
        with open(structured_path, "w", encoding="utf-8") as f:
            json.dump(structured, f, indent=2, ensure_ascii=False)
        print(f"\nStructured JSON saved to: {structured_path}")

        # Step 11: Print summary
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
                f"status={decl.status.value}, value={decl.value}"
            )
        print(f"{'=' * 60}\n")

        return {
            "product_id": product_id,
            "images_processed": len(images),
            "ocr_blocks": len(all_blocks),
            "raw_ocr_path": raw_ocr_path,
            "structured_path": structured_path,
            "extraction": structured,
        }

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
