from .Inspyrenet_Rembg import InspyrenetRembgAdvanced, TextOverlayConfig, TextOverlayBatch, VideoTextOverlay, PersonSelectionPreview

NODE_CLASS_MAPPINGS = {
    "InspyrenetRembgAdvanced": InspyrenetRembgAdvanced,
    "TextOverlayConfig": TextOverlayConfig,
    "TextOverlayBatch": TextOverlayBatch,
    "VideoTextOverlay": VideoTextOverlay,
    "PersonSelectionPreview": PersonSelectionPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InspyrenetRembgAdvanced": "Inspyrenet Rembg Advanced",
    "TextOverlayConfig": "Text Overlay Config",
    "TextOverlayBatch": "Text Overlay Batch",
    "VideoTextOverlay": "Video Text Overlay",
    "PersonSelectionPreview": "Person Selection Preview",
}

__all__ = ['NODE_CLASS_MAPPINGS', "NODE_DISPLAY_NAME_MAPPINGS"]
