"""
Taxonomy Mapper
---------------
Maps Amazon Browse Node / category strings to the fixed 30-label
Shopify-compatible taxonomy defined in data/taxonomy.json.

Mapping strategy:
  - Primary mapping: Amazon top-level category → one or more label IDs
  - Keyword fallback: scan title/description for category signals when
    the top-level category maps to multiple labels (e.g., Clothing_Shoes_and_Jewelry)
  - Unknown categories are logged and skipped (label vector stays all-zeros)

Usage:
    mapper = TaxonomyMapper("data/taxonomy.json")
    labels = mapper.map(amazon_category="Electronics", title="Bluetooth Speaker", description="...")
    # returns list[int] of length 30
"""

import json
import re
from pathlib import Path


# ------------------------------------------------------------------
# Primary Amazon category → label tag mapping
# (many-to-many: one Amazon category can emit multiple labels)
# ------------------------------------------------------------------
AMAZON_CATEGORY_MAP: dict[str, list[str]] = {
    "Electronics":                      ["electronics"],
    "Computers":                        ["computers", "electronics"],
    "Computers&Accessories":            ["computers", "electronics"],
    "Cell_Phones_and_Accessories":      ["phones", "electronics"],
    "Camera_and_Photo":                 ["cameras", "electronics"],
    "Camera_Photo":                     ["cameras", "electronics"],
    "TV_and_Video":                     ["tv-video", "electronics"],
    "Headphones":                       ["audio", "electronics"],
    "Musical_Instruments":              ["music"],
    "Video_Games":                      ["video-games"],
    "Software":                         ["software"],
    # Fashion / apparel — keyword-disambiguated below
    "Clothing_Shoes_and_Jewelry":       ["clothing", "shoes", "jewelry", "bags"],
    "Amazon_Fashion":                   ["clothing", "shoes", "jewelry", "bags"],
    "Luggage_and_Travel_Gear":         ["bags"],
    # Beauty & health
    "All_Beauty":                       ["beauty"],
    "Beauty":                           ["beauty"],
    "Health_and_Household":             ["health"],
    "Health_and_Personal_Care":         ["health"],
    # Home & garden
    "Home_and_Kitchen":                 ["home-kitchen"],
    "Furniture":                        ["furniture", "home-kitchen"],
    "Patio_Lawn_and_Garden":            ["garden"],
    "Tools_and_Home_Improvement":       ["tools-hardware"],
    # Sports & outdoors
    "Sports_and_Outdoors":              ["sports", "outdoors"],
    "Sports":                           ["sports"],
    "Outdoor_Recreation":               ["outdoors"],
    # Automotive
    "Automotive":                       ["automotive"],
    # Toys & baby
    "Toys_and_Games":                   ["toys"],
    "Baby":                             ["baby"],
    "Baby_Products":                    ["baby"],
    # Media
    "Books":                            ["books"],
    "CDs_and_Vinyl":                    ["music"],
    "Digital_Music":                    ["music"],
    "Movies_and_TV":                    ["tv-video"],
    # Office & industrial
    "Office_Products":                  ["office"],
    "Office_and_School_Supplies":       ["office"],
    "Industrial_and_Scientific":        ["industrial"],
    # Other
    "Pet_Supplies":                     ["pet-supplies"],
    "Grocery_and_Gourmet_Food":         ["food-grocery"],
    "Arts_Crafts_and_Sewing":           ["arts-crafts"],
    "Arts_and_Crafts":                  ["arts-crafts"],
    "Appliances":                       ["home-kitchen"],
    "Gift_Cards":                       [],  # no useful label
    "Magazine_Subscriptions":           ["books"],
    "Apps_for_Android":                 ["software"],
    "Prime_Video":                      ["tv-video"],
}

# ------------------------------------------------------------------
# Keyword disambiguation for multi-label Amazon categories
# Maps keyword patterns → single label tag
# Applied when an Amazon category maps to >1 label
# ------------------------------------------------------------------
KEYWORD_DISAMBIGUATION: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bshoe|sneaker|boot|sandal|heel|loafer|slipper\b", re.I), "shoes"),
    (re.compile(r"\bbag|purse|handbag|backpack|tote|wallet|luggage|suitcase\b", re.I), "bags"),
    (re.compile(r"\bnecklace|ring|bracelet|earring|pendant|jewel|watch\b", re.I), "jewelry"),
    (re.compile(r"\bspeaker|headphone|earbud|amplifier|subwoofer|microphone\b", re.I), "audio"),
    (re.compile(r"\btelevision|tv\b|monitor|projector|display\b", re.I), "tv-video"),
    (re.compile(r"\bcamera|lens|tripod|dslr|mirrorless\b", re.I), "cameras"),
    (re.compile(r"\blaptop|desktop|keyboard|mouse|ram|ssd|cpu|gpu|router\b", re.I), "computers"),
    (re.compile(r"\bphone|smartphone|iphone|android|charger|cable\b", re.I), "phones"),
    (re.compile(r"\bsofa|couch|table|chair|desk|shelf|bookcase|bed\b", re.I), "furniture"),
    (re.compile(r"\bsupplement|vitamin|capsule|tablet|pill|protein powder\b", re.I), "health"),
    (re.compile(r"\bcleaner|detergent|spray|wipe|paper towel\b", re.I), "home-kitchen"),
    (re.compile(r"\bdrill|saw|hammer|wrench|screwdriver|level\b", re.I), "tools-hardware"),
    (re.compile(r"\btent|hiking|camping|kayak|bike|bicycle\b", re.I), "outdoors"),
    (re.compile(r"\bjersey|ball|racket|glove|helmet|fitness|gym\b", re.I), "sports"),
]


class TaxonomyMapper:
    def __init__(self, taxonomy_path: str = "data/taxonomy.json"):
        with open(taxonomy_path) as f:
            taxonomy = json.load(f)

        self.labels: list[dict] = taxonomy["labels"]
        self.num_labels: int = len(self.labels)
        self._tag_to_id: dict[str, int] = {l["tag"]: l["id"] for l in self.labels}
        self._unmapped_categories: set[str] = set()

    def map(self, amazon_category: str, title: str = "", description: str = "") -> list[int]:
        """
        Returns a binary label vector of length num_labels.

        Args:
            amazon_category: Top-level Amazon category string
            title:           Product title (used for keyword disambiguation)
            description:     Product description (used for keyword disambiguation)
        """
        vector = [0] * self.num_labels

        # Normalize the category key
        normalized = self._normalize_category(amazon_category)
        candidate_tags = AMAZON_CATEGORY_MAP.get(normalized)

        if candidate_tags is None:
            self._unmapped_categories.add(amazon_category)
            return vector

        if len(candidate_tags) == 0:
            return vector

        if len(candidate_tags) == 1:
            self._set_label(vector, candidate_tags[0])
            return vector

        # Multiple candidate tags — use keyword disambiguation
        text = f"{title} {description}"
        matched = False
        for pattern, tag in KEYWORD_DISAMBIGUATION:
            if tag in candidate_tags and pattern.search(text):
                self._set_label(vector, tag)
                matched = True

        # If no keyword matched, apply all candidate tags
        if not matched:
            for tag in candidate_tags:
                self._set_label(vector, tag)

        return vector

    def map_multiple(self, amazon_categories: list[str], title: str = "", description: str = "") -> list[int]:
        """Union mapping for products with multiple Browse Node categories."""
        vector = [0] * self.num_labels
        for cat in amazon_categories:
            partial = self.map(cat, title, description)
            vector = [max(v, p) for v, p in zip(vector, partial)]
        return vector

    def tag_names(self, vector: list[int]) -> list[str]:
        """Convert a binary label vector to tag name strings."""
        return [self.labels[i]["tag"] for i, v in enumerate(vector) if v == 1]

    def unmapped_report(self) -> set[str]:
        """Returns all Amazon categories that had no taxonomy mapping."""
        return self._unmapped_categories.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_label(self, vector: list[int], tag: str):
        idx = self._tag_to_id.get(tag)
        if idx is not None:
            vector[idx] = 1

    @staticmethod
    def _normalize_category(category: str) -> str:
        return category.strip().replace(" ", "_").replace("&", "and")
