"""
VLM Semantic Declaration Resolver Module.

Uses Qwen-VL (Qwen3-VL-8B via OpenRouter) to semantically resolve
ambiguous packaging declarations, especially:
- brand_name
- common_name / generic_name
- product_description (flavour vs product name, marketing claims)
- ingredient vs product name
- ambiguous declaration-to-value associations

CRITICAL CONSTRAINTS:
- Qwen does NOT perform OCR from scratch. It only classifies/associates
  text that is already recognized in OCR blocks and visible in images.
- No hallucinated/invented text.
- If uncertain: status = "unresolved".
- No legal compliance decisions.
"""

import os
import re
import json
import base64
import logging
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv

from ..models.schemas import OCRBlock, DeclarationCandidate, DeclarationStatus

load_dotenv()

logger = logging.getLogger(__name__)


class VLMSemanticResolver:
    """Semantic declaration resolver powered by Qwen3-VL via OpenRouter."""

    OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODEL = "qwen/qwen3-vl-8b-instruct"
    PRODUCT_CATEGORIES = {
        "food", "beverages", "alcoholic_beverages", "personal_care",
        "cosmetics", "pharmaceuticals", "medical_devices",
        "household_cleaning", "baby_and_childcare", "pet_products",
        "textiles_and_apparel", "footwear", "electronics",
        "electrical_appliances", "stationery_and_office", "toys_and_games",
        "sports_and_fitness", "home_and_kitchen", "automotive",
        "hardware_and_tools", "agricultural_products", "industrial_products",
        "chemical_products", "construction_materials",
        "jewellery_and_accessories", "tobacco_and_related_products", "other",
        "unknown",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 45,
    ):
        self.api_key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("QWEN_API_KEY")
        )
        self.model = (
            model
            or os.getenv("VISION_LLM_MODEL")
            or os.getenv("QWEN_MODEL")
            or self.DEFAULT_MODEL
        )
        self.timeout = timeout

        if not self.api_key:
            logger.warning(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY in .env "
                "to enable VLM semantic resolution."
            )

    @property
    def is_available(self) -> bool:
        """Check if VLM resolver is configured with an API key."""
        return bool(self.api_key)

    def resolve_semantics(
        self,
        image_paths: List[str],
        ocr_blocks: List[OCRBlock],
        candidates: Dict[str, List[DeclarationCandidate]],
    ) -> Dict[str, Any]:
        """Resolve semantic and ambiguous declarations using Qwen-VL.

        Args:
            image_paths: Paths to product packaging images.
            ocr_blocks: Normalized OCR text blocks from all images.
            candidates: Dict of deterministic candidates extracted so far.

        Returns:
            Dict mapping field names to resolved dicts:
            {
                "brand_name": {"status": "resolved"|"unresolved", "value": "...", "confidence": float},
                "product_name": {"status": "resolved"|"unresolved", "value": "..."},
                "common_name": {"status": "resolved"|"unresolved", "value": "...", "confidence": float},
                "product_description": {"status": "resolved"|"unresolved", "value": "..."},
                ...
            }
        """
        if not self.is_available:
            logger.info("VLM resolver skipped (no API key configured).")
            return {}

        valid_image_paths = [p for p in image_paths if os.path.exists(p)]
        if not valid_image_paths:
            logger.warning("No valid image files provided for VLM resolution.")
            return {}

        prompt = self._build_prompt(ocr_blocks, candidates)
        content_items = self._prepare_content_items(valid_image_paths, prompt)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/packaged-commodities/pipeline",
            "X-Title": "Packaged Commodities Declaration Resolver",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content_items,
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
        }

        try:
            logger.info(
                "Sending semantic resolution request to OpenRouter model: %s",
                self.model,
            )
            response = requests.post(
                self.OPENROUTER_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            res_json = response.json()

            choices = res_json.get("choices", [])
            if not choices:
                logger.warning("OpenRouter returned empty choices list.")
                return {}

            raw_text = choices[0].get("message", {}).get("content", "")
            return self._validate_results(
                self._parse_vlm_response(raw_text), ocr_blocks
            )

        except requests.exceptions.RequestException as e:
            logger.error("VLM Semantic Resolver API error: %s", e)
            return {}
        except Exception as e:
            logger.error("Unexpected error in VLM Semantic Resolver: %s", e)
            return {}

    def _prepare_content_items(
        self, image_paths: List[str], prompt: str
    ) -> List[Dict[str, Any]]:
        """Prepare multimodal payload with images and prompt for OpenRouter."""
        items: List[Dict[str, Any]] = []

        # Add image URLs (base64 encoded)
        for idx, path in enumerate(image_paths):
            try:
                mime_type = "image/jpeg"
                ext = os.path.splitext(path)[1].lower()
                if ext == ".png":
                    mime_type = "image/png"
                elif ext == ".webp":
                    mime_type = "image/webp"

                with open(path, "rb") as f:
                    b64_data = base64.b64encode(f.read()).decode("utf-8")

                items.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{b64_data}",
                    },
                })
            except Exception as e:
                logger.warning("Failed to encode image [%s]: %s", path, e)

        # Add text prompt
        items.append({
            "type": "text",
            "text": prompt,
        })

        return items

    def _build_prompt(
        self,
        ocr_blocks: List[OCRBlock],
        candidates: Dict[str, List[DeclarationCandidate]],
    ) -> str:
        """Build precise instructions for semantic classification."""
        ocr_lines_summary = []
        for b in ocr_blocks:
            ocr_lines_summary.append(
                f"- [img_{b.image_index}] '{b.text}' (bbox: {b.bbox})"
            )
        ocr_text_repr = "\n".join(ocr_lines_summary)

        # Summarize current extraction status
        cand_summary = []
        for field, c_list in candidates.items():
            if c_list:
                cand_texts = [
                    f"value='{c.value}', status={c.status.value}, type={c.metadata.get('type')}"
                    for c in c_list
                ]
                cand_summary.append(f"- {field}: {'; '.join(cand_texts)}")
            else:
                cand_summary.append(f"- {field}: not_detected")
        cand_text_repr = "\n".join(cand_summary)

        prompt = f"""You are a specialized Packaging Declaration Classifier for packaged goods.
You are inspecting the attached product packaging image(s) alongside the OCR text blocks extracted from them.

DETECTED OCR TEXT BLOCKS:
{ocr_text_repr}

CURRENT CANDIDATE EXTRACTIONS:
{cand_text_repr}

YOUR TASK:
Determine the exact semantic classifications for declarations on this package. Specifically:

IMPORTANT MULTI-BLOCK RULE:
- A single semantic field may span multiple consecutive OCR blocks, including blocks
    from the same image or across product images.
- Use image content, image_index, bbox coordinates, reading order, and nearby OCR
    context to identify continuation blocks.
- When blocks form one phrase, concatenate their OCR wording in reading order,
    preserving the exact words, punctuation, and spelling shown by OCR.
- Never truncate, paraphrase, invent, or omit meaningful words. Do not replace OCR
    wording with a guessed correction. For example, if OCR says `Vitamin B,`,
    return `Vitamin B,`, never `Vitamin B6`.

1. `brand_name`:
   - Identify the overarching brand or trade name (e.g. "CORONA", "Amul", "Britannia", "Nestlé", "Tata").
   - Distinguish brand from the product/formulation name, marketing claims, and generic descriptions.
   - If not clearly distinguishable, set status = "unresolved" and value = null.

2. `common_name` (or generic name):
   - Identify the common, generic, or statutory description of what the product actually is (e.g. "Ice Cream", "Potassium Citrate, Magnesium Citrate and Vitamin B6 Oral Solution", "Wheat Flour", "Biscuits", "Shampoo").
   - DO NOT mistake flavour descriptions (e.g., "Pineapple Flavour", "Vanilla", "Masala") or marketing claims (e.g., "Sugar Free", "100% Natural", "Delicious") for the common name.
   - If not clearly identifiable, set status = "unresolved" and value = null.

3. `product_description`:
    - Flavour description, marketing claims, or supplementary formulation subtitles (e.g. "Delicious Pineapple Flavour, Sugar Free").
    - Join all relevant adjacent OCR blocks into the complete phrase.

4. `product_name`:
    - Identify the specific product/trade name, such as "Alka-Shot CP".
    - Resolve only when semantically confident and select only visible OCR text.
    - Otherwise return status = "unresolved" and value = null.

5. For warnings, instructions, formulations, ingredients, and other semantic
   phrases, apply the same multi-block joining rule and return the complete OCR
   phrase when the association is clear.

6. Disambiguate any ambiguous fields:
   - If multiple candidates or unclear associations exist for `net_quantity`, `mrp`, `manufacture_pack_import_date`, `best_before_use_by`, or `manufacturer_packer_importer`, use the visual packaging context to resolve the exact correct value.
     - For manufacture/packing/import dates, read the explicit date next to labels
         such as `MFG`, `PKD`, `Manufactured`, or `Packed`.
     - For expiry/use-by dates, read the explicit date next to `USE BY`, `BEST BEFORE`,
         `EXPIRY`, or equivalent labels.
     - Dates may be day/month/year, month/year, or an explicit written month/year.
         Never classify an unexplained compact numeric code such as `20222257` as a date.

CRITICAL RULES:
- ONLY select text that is ACTUALLY VISIBLE on the provided packaging image(s) and present in the OCR blocks.
- DO NOT invent, guess, or hallucinate missing text.
- If a declaration cannot be determined reliably from the image, set status = "unresolved" and value = null.
- Do NOT make legal compliance decisions.
- Output MUST be valid JSON only.

5. `product_classification`:
     - Classify the product using exactly one category from this fixed list:
         food, beverages, alcoholic_beverages, personal_care, cosmetics,
         pharmaceuticals, medical_devices, household_cleaning, baby_and_childcare,
         pet_products, textiles_and_apparel, footwear, electronics,
         electrical_appliances, stationery_and_office, toys_and_games,
         sports_and_fitness, home_and_kitchen, automotive, hardware_and_tools,
         agricultural_products, industrial_products, chemical_products,
         construction_materials, jewellery_and_accessories,
         tobacco_and_related_products, other, unknown.
     - Use only visible packaging context and OCR text. If uncertain, use unknown.
     - Return confidence as a number from 0.0 to 1.0.

RESPOND ONLY WITH A JSON OBJECT matching this format:
```json
{{
    "product_classification": {{
        "category": "food",
        "confidence": 0.95
    }},
  "brand_name": {{
    "status": "resolved",
    "value": "CORONA"
  }},
    "product_name": {{
        "status": "resolved",
        "value": "Alka-Shot CP"
    }},
  "common_name": {{
    "status": "resolved",
    "value": "Potassium Citrate, Magnesium Citrate and Vitamin B6 Oral Solution"
  }},
  "product_description": {{
    "status": "resolved",
    "value": "Delicious Pineapple Flavour, Sugar Free"
  }},
  "net_quantity": {{
    "status": "resolved",
    "value": "450 ml"
  }},
  "mrp": {{
    "status": "resolved",
    "value": 373.17
  }},
  "manufacturer_packer_importer": {{
    "status": "resolved",
    "value": "CORONA Remedies Limited"
  }},
  "manufacture_pack_import_date": {{
    "status": "resolved",
    "value": "05/2026"
  }},
  "best_before_use_by": {{
    "status": "resolved",
    "value": "10/2027"
  }},
  "country_of_origin": {{
    "status": "not_detected",
    "value": null
  }},
  "consumer_care": {{
    "status": "not_detected",
    "value": null
  }},
  "unit_sale_price": {{
    "status": "not_detected",
    "value": null
  }},
  "dimensions": {{
    "status": "not_detected",
    "value": null
  }}
}}
```
"""
        return prompt

    def _parse_vlm_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse and validate JSON response from Qwen."""
        if not raw_text:
            return {}

        text = raw_text.strip()
        # Remove markdown code blocks if present
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced_match:
            text = fenced_match.group(1).strip()
        else:
            brace_match = re.search(r"\{.*\}", text, re.DOTALL)
            if brace_match:
                text = brace_match.group(0).strip()

        try:
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                logger.warning("VLM response was not a JSON dict: %s", raw_text)
                return {}

            logger.info("VLM semantic resolution successfully parsed %d fields", len(parsed))
            return parsed
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse VLM response as JSON: %s (error: %s)", raw_text, e)
            return {}

    def _validate_classification(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the fixed-category Qwen classification response."""
        classification = results.get("product_classification")
        if not isinstance(classification, dict):
            return {"category": "unknown", "confidence": 0.0}

        category = classification.get("category")
        if category not in self.PRODUCT_CATEGORIES:
            return {"category": "unknown", "confidence": 0.0}
        try:
            confidence = float(classification.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "category": category,
            "confidence": max(0.0, min(1.0, confidence)),
        }

    def _validate_results(
        self, results: Dict[str, Any], ocr_blocks: List[OCRBlock]
    ) -> Dict[str, Any]:
        """Keep only schema-valid values that occur in the OCR text."""
        allowed_fields = {
            "brand_name", "product_name", "common_name", "product_description",
            "manufacturer_packer_importer", "country_of_origin",
            "net_quantity", "mrp", "manufacture_pack_import_date",
            "best_before_use_by", "consumer_care", "unit_sale_price",
            "dimensions",
        }
        visible_text = " ".join(block.text for block in ocr_blocks).casefold()
        validated: Dict[str, Any] = {}

        validated["product_classification"] = self._validate_classification(results)

        for field, data in results.items():
            if field not in allowed_fields or not isinstance(data, dict):
                continue
            status = str(data.get("status", "")).lower()
            value = data.get("value")
            if status == "unresolved":
                validated[field] = {"status": "unresolved", "value": None}
                continue
            if status != "resolved" or value is None:
                continue
            if field in {
                "manufacture_pack_import_date",
                "best_before_use_by",
            } and not self._is_explicit_date_value(value):
                logger.warning(
                    "Discarding non-date value for %s: %r", field, value
                )
                validated[field] = {"status": "unresolved", "value": None}
                continue
            value_text = str(value).casefold().strip()
            if value_text and self._value_is_visible(value_text, ocr_blocks):
                validated[field] = {"status": "resolved", "value": value}
            else:
                reconstructed = self._reconstruct_ocr_phrase(value_text, ocr_blocks)
                if reconstructed is not None:
                    validated[field] = {
                        "status": "resolved",
                        "value": reconstructed,
                    }
                    continue
                logger.warning(
                    "Discarding Qwen value for %s because it is absent from OCR blocks: %r",
                    field, value,
                )
                validated[field] = {"status": "unresolved", "value": None}

        return validated

    @staticmethod
    def _is_explicit_date_value(value: Any) -> bool:
        """Reject unexplained numeric codes from date declarations."""
        text = str(value).strip()
        if re.fullmatch(r"\d{6,}", text):
            return False
        return bool(
            re.search(r"\b\d{1,2}\s*[/.-]\s*\d{1,2}\s*[/.-]\s*\d{2,4}\b", text)
            or re.search(r"\b\d{1,2}\s*[/.-]\s*\d{4}\b", text)
            or re.search(
                r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
                r"\.?\s*[, -]?\s*\d{2,4}\b",
                text,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _reconstruct_ocr_phrase(
        value: Any, ocr_blocks: List[OCRBlock]
    ) -> Optional[str]:
        """Reconstruct a semantic phrase from consecutive OCR blocks exactly."""
        value_tokens = re.findall(r"[\w]+", str(value).casefold())
        if len(value_tokens) < 2:
            return None

        normalized_blocks = [
            re.findall(r"[\w]+", block.text.casefold()) for block in ocr_blocks
        ]
        best_phrase = None
        best_score = 0.0
        best_length = None
        for start in range(len(ocr_blocks)):
            phrase_tokens: List[str] = []
            for end in range(start, len(ocr_blocks)):
                phrase_tokens.extend(normalized_blocks[end])
                if not phrase_tokens:
                    continue
                matched = sum(
                    1 for token in value_tokens if token in phrase_tokens
                )
                recall = matched / len(value_tokens)
                precision = matched / len(phrase_tokens)
                score = (
                    2 * recall * precision / (recall + precision)
                    if recall and precision
                    else 0.0
                )
                if score > best_score and score >= 0.8:
                    best_score = score
                    best_length = len(phrase_tokens)
                    best_phrase = " ".join(
                        block.text.strip() for block in ocr_blocks[start:end + 1]
                    )
                elif (
                    score == best_score
                    and score >= 0.8
                    and (best_length is None or len(phrase_tokens) < best_length)
                ):
                    best_length = len(phrase_tokens)
                    best_phrase = " ".join(
                        block.text.strip() for block in ocr_blocks[start:end + 1]
                    )
                if len(phrase_tokens) > len(value_tokens) + 3:
                    break
        return best_phrase

    @staticmethod
    def _value_is_visible(value: str, ocr_blocks: List[OCRBlock]) -> bool:
        """Accept exact OCR wording or a visible multi-block token set."""
        def normalize(text: str) -> str:
            return re.sub(r"\s+", " ", text.casefold()).strip()

        normalized_value = normalize(value)
        normalized_blocks = [normalize(block.text) for block in ocr_blocks]
        if normalized_value in " ".join(normalized_blocks):
            return True
        for start in range(len(normalized_blocks)):
            joined = ""
            for block_text in normalized_blocks[start:]:
                joined = f"{joined} {block_text}".strip()
                if joined == normalized_value:
                    return True
                if len(joined) > len(normalized_value):
                    break

        value_tokens = re.findall(r"[\w]+", normalized_value)
        visible_tokens = re.findall(
            r"[\w]+", " ".join(normalized_blocks)
        )
        remaining = list(visible_tokens)
        for token in value_tokens:
            if token not in remaining:
                return False
            remaining.remove(token)
        return bool(value_tokens)
