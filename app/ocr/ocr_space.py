"""
OCR.Space API adapter module.

Sends images to OCR.Space API and returns normalized OCR blocks.
This is the ONLY OCR engine in this pipeline version.

The adapter is responsible for:
- Sending images to OCR.Space
- Handling API errors gracefully
- Normalizing responses into engine-independent OCR blocks

It does NOT:
- Identify brands, MRP, or any declarations
- Preprocess images
- Make compliance decisions
"""

import os
import json
import logging
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class OCRSpaceAdapter:
    """Adapter for OCR.Space API.
    
    Converts OCR.Space responses into normalized OCR blocks that
    downstream extractors can consume independently of the OCR provider.
    """

    ENGINE_NAME = "ocr_space"
    API_URL = "https://api.ocr.space/parse/image"
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp", ".pdf"}

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        """Initialize the OCR.Space adapter.
        
        Args:
            api_key: OCR.Space API key. Falls back to OCR_SPACE_API_KEY env var.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or os.getenv("OCR_SPACE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OCR.Space API key not found. "
                "Set OCR_SPACE_API_KEY in .env or pass it directly."
            )
        self.timeout = timeout

    def process_image(self, image_path: str, image_index: int = 0) -> Dict[str, Any]:
        """Send a single image to OCR.Space and return the raw response.
        
        Args:
            image_path: Path to the image file.
            image_index: Index of the image in the batch (for multi-image products).
            
        Returns:
            Dict containing:
                - image_path: Original path
                - image_index: Image index
                - success: Whether OCR succeeded
                - error: Error message if failed, None if success
                - error_type: Categorized error type (network, timeout, api_error, parse_error, file_not_found)
                - raw_response: Complete OCR.Space API response
        """
        # Validate file exists
        if not os.path.exists(image_path):
            return self._error_result(
                image_path, image_index,
                f"Image file not found: {image_path}",
                "file_not_found"
            )

        # Validate extension
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            return self._error_result(
                image_path, image_index,
                f"Unsupported file extension: {ext}",
                "unsupported_format"
            )

        try:
            with open(image_path, "rb") as image_file:
                response = requests.post(
                    self.API_URL,
                    headers={"apikey": self.api_key},
                    files={"file": image_file},
                    data={
                        "language": "eng",
                        "OCREngine": "3",
                        "isOverlayRequired": "true",
                        "detectOrientation": "true",
                        "scale": "true",
                    },
                    timeout=self.timeout,
                )

            response.raise_for_status()
            result = response.json()

            # Check OCR.Space-level errors
            ocr_exit_code = result.get("OCRExitCode", -1)
            is_errored = result.get("IsErroredOnProcessing", True)

            if ocr_exit_code != 1 or is_errored:
                error_msgs = result.get("ErrorMessage", ["Unknown OCR error"])
                if isinstance(error_msgs, list):
                    error_text = "; ".join(str(m) for m in error_msgs)
                else:
                    error_text = str(error_msgs)
                return self._error_result(
                    image_path, image_index,
                    f"OCR.Space processing error: {error_text}",
                    "api_error",
                    raw_response=result,
                )

            logger.info(
                "OCR.Space successfully processed image [%d]: %s",
                image_index, os.path.basename(image_path),
            )

            return {
                "image_path": image_path,
                "image_index": image_index,
                "success": True,
                "error": None,
                "error_type": None,
                "raw_response": result,
            }

        except requests.exceptions.Timeout:
            return self._error_result(
                image_path, image_index,
                f"OCR.Space API request timed out after {self.timeout}s",
                "timeout",
            )
        except requests.exceptions.ConnectionError as e:
            return self._error_result(
                image_path, image_index,
                f"Network error connecting to OCR.Space: {e}",
                "network",
            )
        except requests.exceptions.HTTPError as e:
            return self._error_result(
                image_path, image_index,
                f"OCR.Space API HTTP error: {e}",
                "api_error",
            )
        except requests.exceptions.RequestException as e:
            return self._error_result(
                image_path, image_index,
                f"OCR.Space API request failed: {e}",
                "network",
            )
        except json.JSONDecodeError:
            return self._error_result(
                image_path, image_index,
                "Failed to parse OCR.Space response as JSON",
                "parse_error",
            )

    def normalize_blocks(
        self, raw_response: Dict[str, Any], image_index: int = 0
    ) -> List[Dict[str, Any]]:
        """Normalize OCR.Space response into standard OCR blocks.
        
        Extracts line-level text blocks with bounding boxes from the
        OCR.Space overlay data. Each block contains:
        - text: The recognized text for the line
        - bbox: [x, y, width, height] computed from word bboxes
        - confidence: null (OCR.Space Engine 3 does not provide confidence)
        - image_index: Which image this block came from
        - source_engine: Always "ocr_space"
        - words: Individual word details with text and bbox
        
        Args:
            raw_response: Complete OCR.Space API response dict.
            image_index: Index of the source image.
            
        Returns:
            List of normalized OCR block dictionaries.
        """
        blocks: List[Dict[str, Any]] = []

        if not raw_response:
            return blocks

        parsed_results = raw_response.get("ParsedResults", [])

        for parsed in parsed_results:
            overlay = parsed.get("TextOverlay", {})
            lines = overlay.get("Lines", [])

            for line in lines:
                line_text = line.get("LineText", "").strip()
                if not line_text:
                    continue

                words = line.get("Words", [])
                line_bbox = self._compute_line_bbox(words)

                word_details = [
                    {
                        "text": w.get("WordText", ""),
                        "bbox": [
                            w.get("Left", 0),
                            w.get("Top", 0),
                            w.get("Width", 0),
                            w.get("Height", 0),
                        ],
                    }
                    for w in words
                ]

                blocks.append({
                    "text": line_text,
                    "bbox": line_bbox,
                    "confidence": None,  # OCR.Space Engine 3 does not provide confidence
                    "image_index": image_index,
                    "source_engine": self.ENGINE_NAME,
                    "words": word_details,
                })

        logger.info(
            "Normalized %d OCR blocks from image [%d]",
            len(blocks), image_index,
        )
        return blocks

    def _compute_line_bbox(self, words: List[Dict]) -> List[float]:
        """Compute bounding box for a line from its word bounding boxes.
        
        Finds the minimum enclosing rectangle across all words in the line.
        
        Args:
            words: List of word dicts from OCR.Space with Left, Top, Width, Height.
            
        Returns:
            [x, y, width, height] for the entire line.
        """
        if not words:
            return [0, 0, 0, 0]

        min_left = min(w.get("Left", 0) for w in words)
        min_top = min(w.get("Top", 0) for w in words)
        max_right = max(w.get("Left", 0) + w.get("Width", 0) for w in words)
        max_bottom = max(w.get("Top", 0) + w.get("Height", 0) for w in words)

        return [min_left, min_top, max_right - min_left, max_bottom - min_top]

    @staticmethod
    def _error_result(
        image_path: str,
        image_index: int,
        error: str,
        error_type: str,
        raw_response: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Create a standardized error result dict."""
        logger.warning(
            "OCR error [%s] for image [%d] %s: %s",
            error_type, image_index, os.path.basename(image_path), error,
        )
        return {
            "image_path": image_path,
            "image_index": image_index,
            "success": False,
            "error": error,
            "error_type": error_type,
            "raw_response": raw_response,
        }
