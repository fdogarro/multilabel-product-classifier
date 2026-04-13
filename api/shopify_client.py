"""
Shopify Admin API Client
-------------------------
PATCH /products/:id — writes predicted tags back to a Shopify product.
Handles webhook events: products/create and products/update.
Implemented in Phase 6.
"""


class ShopifyClient:
    def __init__(self, store_url: str, access_token: str):
        self.store_url = store_url.rstrip("/")
        self.access_token = access_token

    def update_tags(self, product_id: str, tags: list[str]) -> dict:
        raise NotImplementedError("Phase 6")

    def handle_webhook(self, payload: dict) -> dict:
        raise NotImplementedError("Phase 6")
