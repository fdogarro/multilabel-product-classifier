"""
FastAPI Application
-------------------
POST /tag  — accepts product JSON, returns tag vector + routing metadata
Implemented in Phase 6.
"""
from fastapi import FastAPI

app = FastAPI(title="Shopify Tagger", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/tag")
def tag_product(product: dict) -> dict:
    """
    Request body:
      { "title": str, "description": str, "vendor"?: str, "product_type"?: str }

    Response:
      { "tags": list[str], "confidence": list[float],
        "model_used": str, "routed_by": str, "flagged": bool }
    """
    raise NotImplementedError("Phase 6")
