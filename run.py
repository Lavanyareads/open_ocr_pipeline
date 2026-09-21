"""
Entry point for the OCR.Space extraction pipeline.

Usage:
    python run.py
    python run.py --input-dir my_images --product-id my_product
    python run.py --output-dir results
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

# Ensure Windows stdout/stderr handles Unicode symbols (e.g. ₹ rupee symbol)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load environment variables before importing pipeline
load_dotenv()

from app.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(
        description="OCR.Space Extraction Pipeline for Packaged Commodities"
    )
    parser.add_argument(
        "--input-dir",
        default="input_images",
        help="Directory containing input images (default: input_images)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for output files (default: output)",
    )
    parser.add_argument(
        "--product-id",
        default=None,
        help="Product identifier (auto-generated if not provided)",
    )
    parser.add_argument(
        "--barcode-width-mm",
        type=float,
        default=None,
        help="Optional physical barcode width in millimetres for scale calibration",
    )
    parser.add_argument(
        "--barcode-height-mm",
        type=float,
        default=None,
        help="Optional physical barcode height in millimetres for scale calibration",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        pipeline = Pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            barcode_width_mm=args.barcode_width_mm,
            barcode_height_mm=args.barcode_height_mm,
        )
    except ValueError as e:
        print(f"\nConfiguration error: {e}")
        sys.exit(1)

    result = pipeline.run(product_id=args.product_id)

    if "error" in result:
        print(f"\nPipeline completed with errors: {result['error']}")
        sys.exit(1)

    results = result.get("products", [result])
    print(f"\nPipeline completed successfully for {len(results)} product(s)!")
    for product_result in results:
        structured = product_result.get("extraction", {}).get("declarations", {})
        resolved = sum(1 for item in structured.values() if item.get("status") == "resolved")
        unresolved = sum(1 for item in structured.values() if item.get("status") == "unresolved")
        print(f"  Product: {product_result['product_id']}")
        print(f"    Images processed: {product_result.get('images_processed', 'see raw OCR')}")
        print(f"    OCR blocks detected: {product_result.get('ocr_blocks', 'see raw OCR')}")
        print(f"    Declarations resolved: {resolved}")
        print(f"    Declarations unresolved: {unresolved}")
        parties = structured.get("manufacturer_packer_importer", {}).get("entries", [])
        for party in parties:
            print(f"    {party['role']}={party['value']}")
        print(f"    Structured: {product_result['structured_path']}")


if __name__ == "__main__":
    main()
