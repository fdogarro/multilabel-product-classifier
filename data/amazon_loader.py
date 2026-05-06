"""
Amazon Product Loader
---------------------
Loads product metadata from the Amazon Product Reviews 2023 dataset
(McAuley Lab, UC San Diego) via HuggingFace Datasets.

Target categories are selected to cover all 30 taxonomy labels while
keeping the training corpus to ~80K products.

Usage:
    loader = AmazonLoader(taxonomy_mapper, cache_dir="data/raw/amazon")
    df = loader.load(max_products=80_000)
    df.to_parquet("data/processed/amazon_raw.parquet", index=False)
"""

import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset, concatenate_datasets

from data.taxonomy_mapper import TaxonomyMapper

logger = logging.getLogger(__name__)

# Categories to pull, chosen to cover all 30 taxonomy labels.
# HuggingFace dataset name: McAuley-Lab/Amazon-Reviews-2023
# Config format: "raw_meta_{Category}"
TARGET_CATEGORIES = [
    "All_Beauty",
    "Toys_and_Games",
    "Cell_Phones_and_Accessories",
    "Industrial_and_Scientific",
    "Musical_Instruments",
    "Electronics",
    "Clothing_Shoes_and_Jewelry",
    "Arts_Crafts_and_Sewing",
    "Office_Products",
    "Home_and_Kitchen",
    "Sports_and_Outdoors",
    "Automotive",
    "Video_Games",
    "Pet_Supplies",
    "Software",
    "Health_and_Household",
    "Patio_Lawn_and_Garden",
    "Tools_and_Home_Improvement",
    "Books",
    "Grocery_and_Gourmet_Food",
    "Baby_Products",
]

DATASET_REPO = "McAuley-Lab/Amazon-Reviews-2023"


class AmazonLoader:
    def __init__(self, taxonomy_mapper: TaxonomyMapper, cache_dir: str = "data/raw/amazon"):
        self.mapper = taxonomy_mapper
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, max_products: int = 80_000, categories: list[str] = None) -> pd.DataFrame:
        """
        Download and process Amazon product metadata.

        Args:
            max_products: Total cap across all categories
            categories:   Override TARGET_CATEGORIES if provided

        Returns:
            DataFrame with columns: title, description, vendor,
            product_type, amazon_category, labels (list[int])
        """
        cats = categories or TARGET_CATEGORIES
        per_cat = max(1, max_products // len(cats))
        records = []

        for cat in cats:
            logger.info(f"Loading {cat} (up to {per_cat} products)...")
            try:
                ds = load_dataset(
                    DATASET_REPO,
                    f"raw_meta_{cat}",
                    split="full",
                    streaming=True,
                )
                records.extend(self._process_category(ds, cat, per_cat))
            except Exception as e:
                logger.warning(f"Failed to load {cat}: {e}")
                continue

        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df):,} products across {len(cats)} categories")

        # Log unmapped categories
        unmapped = self.mapper.unmapped_report()
        if unmapped:
            logger.warning(f"Unmapped Amazon categories: {unmapped}")

        return df

    def _process_category(self, dataset, category: str, limit: int) -> list[dict]:
        records = []
        for item in dataset:
            if len(records) >= limit:
                break

            title = item.get("title") or ""
            # description may be a list or a string depending on the category
            desc_raw = item.get("description") or ""
            if isinstance(desc_raw, list):
                description = " ".join(str(d) for d in desc_raw if d)
            else:
                description = str(desc_raw)

            # Skip products with no usable text
            if not title.strip() and not description.strip():
                continue

            # Map categories to label vector
            categories_field = item.get("categories") or []
            if isinstance(categories_field, list) and categories_field:
                labels = self.mapper.map_multiple(categories_field, title, description)
            else:
                labels = self.mapper.map(category, title, description)

            # Skip if no label could be assigned
            if sum(labels) == 0:
                continue

            records.append({
                "title": title,
                "description": description,
                "vendor": item.get("brand") or item.get("vendor") or "",
                "product_type": category,
                "amazon_category": category,
                "asin": item.get("parent_asin") or item.get("asin") or "",
                "labels": labels,
            })

        return records
