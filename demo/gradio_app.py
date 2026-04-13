"""
Gradio Demo
-----------
Interactive demo: title + description inputs → predicted tags,
confidence scores, and routing decision visualization.
Implemented in Phase 6.
"""
import gradio as gr


def predict(title: str, description: str) -> tuple:
    """Stub — wired to the full pipeline in Phase 6."""
    raise NotImplementedError("Phase 6")


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Product Title"),
        gr.Textbox(label="Product Description", lines=4),
    ],
    outputs=[
        gr.Label(label="Predicted Tags"),
        gr.JSON(label="Routing Decision"),
    ],
    title="Shopify Product Tagger",
    description="Multi-agent LLM pipeline for intelligent product tagging.",
)

if __name__ == "__main__":
    demo.launch()
