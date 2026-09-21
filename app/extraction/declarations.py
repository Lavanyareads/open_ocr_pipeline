"""
Declaration resolution module.

Takes candidates from deterministic extraction, spatial association,
and VLM semantic resolution, resolving final values for each declaration
field into a ProductExtraction model.
"""

from typing import Dict, List, Any, Optional
from ..models.schemas import (
    DeclarationCandidate,
    DeclarationValue,
    DeclarationStatus,
    ProductExtraction,
)


class DeclarationResolver:
    """Resolves final declaration values from candidates and VLM semantic resolutions."""

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

    def resolve(
        self,
        product_id: str,
        candidates: Dict[str, List[DeclarationCandidate]],
        vlm_results: Optional[Dict[str, Any]] = None,
    ) -> ProductExtraction:
        """Resolve candidates and VLM output into final declaration values.

        Args:
            product_id: Unique identifier for the product.
            candidates: Dict of field_name -> list of deterministic candidates.
            vlm_results: Dict of field_name -> VLM semantic result dict (optional).

        Returns:
            ProductExtraction with resolved declarations.
        """
        vlm_results = vlm_results or {}
        declarations: Dict[str, DeclarationValue] = {}

        for field in self.DECLARATION_FIELDS:
            field_candidates = candidates.get(field, [])
            vlm_field_data = vlm_results.get(field)
            declarations[field] = self._resolve_field(
                field, field_candidates, vlm_field_data
            )

        return ProductExtraction(
            product_id=product_id,
            declarations=declarations,
        )

    def _resolve_field(
        self,
        field_name: str,
        candidates: List[DeclarationCandidate],
        vlm_data: Optional[Dict[str, Any]] = None,
    ) -> DeclarationValue:
        """Resolve a single declaration field."""
        # 1. Check if deterministic extraction already produced a strong resolved candidate
        sorted_candidates = sorted(
            candidates, key=lambda c: c.confidence_score, reverse=True
        )

        source_images = sorted(list(set(
            block.image_index
            for c in candidates
            for block in c.source_blocks
        )))

        candidate_dicts = [
            {
                "value": c.value,
                "raw_text": c.raw_text,
                "confidence_score": c.confidence_score,
                "type": c.metadata.get("type", "unknown"),
                "source_images": sorted(list(set(b.image_index for b in c.source_blocks))),
            }
            for c in sorted_candidates
        ]

        best_deterministic = next(
            (c for c in sorted_candidates if c.status == DeclarationStatus.RESOLVED and c.value is not None),
            None
        )

        # 2. For semantic fields (brand_name, common_name), prioritize VLM if available
        is_semantic_field = field_name in ("brand_name", "common_name")

        if is_semantic_field and vlm_data:
            vlm_status_str = str(vlm_data.get("status", "")).lower()
            vlm_val = vlm_data.get("value")

            if vlm_status_str == "resolved" and vlm_val:
                metadata = {"resolution_source": "qwen3_vl_semantic_resolver"}
                return DeclarationValue(
                    value=vlm_val,
                    status=DeclarationStatus.RESOLVED,
                    raw_text=str(vlm_val),
                    source_images=source_images,
                    metadata=metadata,
                )
            elif vlm_status_str == "unresolved":
                return DeclarationValue(
                    value=None,
                    status=DeclarationStatus.UNRESOLVED,
                    raw_text=str(vlm_val) if vlm_val else None,
                    source_images=source_images,
                    candidates=candidate_dicts,
                    metadata={"resolution_source": "qwen3_vl_semantic_resolver"},
                )

        # 3. If deterministic has a clean resolved candidate, use it
        if best_deterministic:
            metadata = {k: v for k, v in best_deterministic.metadata.items() if k not in ("type", "matched_text")}
            return DeclarationValue(
                value=best_deterministic.value,
                status=DeclarationStatus.RESOLVED,
                raw_text=best_deterministic.raw_text,
                source_images=source_images,
                candidates=candidate_dicts if len(sorted_candidates) > 1 else [],
                metadata=metadata,
            )

        # 4. If deterministic was unresolved/empty, check if VLM resolved it
        if vlm_data:
            vlm_status_str = str(vlm_data.get("status", "")).lower()
            vlm_val = vlm_data.get("value")

            if vlm_status_str == "resolved" and vlm_val is not None:
                return DeclarationValue(
                    value=vlm_val,
                    status=DeclarationStatus.RESOLVED,
                    raw_text=str(vlm_val),
                    source_images=source_images,
                    candidates=candidate_dicts,
                    metadata={"resolution_source": "qwen3_vl_semantic_resolver"},
                )
            elif vlm_status_str == "unresolved":
                return DeclarationValue(
                    value=vlm_val,
                    status=DeclarationStatus.UNRESOLVED,
                    raw_text=str(vlm_val) if vlm_val else None,
                    source_images=source_images,
                    candidates=candidate_dicts,
                    metadata={"resolution_source": "qwen3_vl_semantic_resolver"},
                )

        # 5. If candidates exist but none resolved
        if sorted_candidates:
            top_candidate = sorted_candidates[0]
            return DeclarationValue(
                value=top_candidate.value,
                status=DeclarationStatus.UNRESOLVED,
                raw_text=top_candidate.raw_text,
                source_images=source_images,
                candidates=candidate_dicts,
                metadata=top_candidate.metadata,
            )

        # 6. Not detected
        return DeclarationValue(
            status=DeclarationStatus.NOT_DETECTED,
        )
