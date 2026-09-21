"""Barcode-based pixel-to-millimetre calibration."""

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import cv2

from ..models.schemas import OCRBlock

logger = logging.getLogger(__name__)


class BarcodeCalibrator:
    """Detect a barcode and estimate OCR text heights in physical units."""

    def __init__(
        self,
        barcode_width_mm: Optional[float] = None,
        barcode_height_mm: Optional[float] = None,
    ):
        self.barcode_width_mm = self._validate_dimension(
            barcode_width_mm, "barcode_width_mm"
        )
        self.barcode_height_mm = self._validate_dimension(
            barcode_height_mm, "barcode_height_mm"
        )

    @staticmethod
    def _validate_dimension(value: Optional[float], name: str) -> Optional[float]:
        if value is None:
            return None
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive number") from exc
        if numeric_value <= 0:
            raise ValueError(f"{name} must be a positive number")
        return numeric_value

    def measure(
        self, image_paths: Sequence[str], ocr_blocks: List[OCRBlock]
    ) -> Dict[str, Any]:
        """Detect the first usable barcode and measure all OCR block heights."""
        detection = self._detect_barcode(image_paths)
        pixels_per_mm = None
        calibration_dimension = None

        if detection["barcode_detected"]:
            if self.barcode_width_mm is not None:
                pixels_per_mm = detection["barcode_width_px"] / self.barcode_width_mm
                calibration_dimension = "width"
            elif self.barcode_height_mm is not None:
                pixels_per_mm = detection["barcode_height_px"] / self.barcode_height_mm
                calibration_dimension = "height"

        text_bboxes = []
        for block in ocr_blocks:
            bbox_height_px = float(block.bbox[3]) if len(block.bbox) > 3 else 0.0
            estimated_height_mm = (
                bbox_height_px / pixels_per_mm
                if pixels_per_mm and bbox_height_px >= 0
                else None
            )
            text_bboxes.append({
                "image_index": block.image_index,
                "text": block.text,
                "bbox": block.bbox,
                "bbox_height_px": bbox_height_px,
                "estimated_height_mm": estimated_height_mm,
            })

        return {
            "barcode_detected": detection["barcode_detected"],
            "barcode_image_index": detection["image_index"],
            "barcode_bbox_px": detection["bbox_px"],
            "barcode_width_px": detection["barcode_width_px"],
            "barcode_height_px": detection["barcode_height_px"],
            "barcode_width_mm": self.barcode_width_mm,
            "barcode_height_mm": self.barcode_height_mm,
            "calibration_dimension": calibration_dimension,
            "pixels_per_mm": pixels_per_mm,
            "text_bboxes": text_bboxes,
        }

    def _detect_barcode(self, image_paths: Sequence[str]) -> Dict[str, Any]:
        detector = cv2.barcode_BarcodeDetector()
        for image_index, image_path in enumerate(image_paths):
            if not os.path.exists(image_path):
                continue
            image = cv2.imread(image_path)
            if image is None:
                continue
            try:
                detected, points = detector.detect(image)
            except cv2.error as exc:
                logger.warning("Barcode detection failed for %s: %s", image_path, exc)
                continue
            if not detected or points is None:
                continue

            polygon = points[0] if len(points.shape) == 3 else points
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            bbox = {
                "x": min(xs),
                "y": min(ys),
                "width": max(xs) - min(xs),
                "height": max(ys) - min(ys),
            }
            if bbox["width"] <= 0 or bbox["height"] <= 0:
                continue
            return {
                "barcode_detected": True,
                "image_index": image_index,
                "bbox_px": bbox,
                "barcode_width_px": bbox["width"],
                "barcode_height_px": bbox["height"],
            }

        return {
            "barcode_detected": False,
            "image_index": None,
            "bbox_px": None,
            "barcode_width_px": None,
            "barcode_height_px": None,
        }