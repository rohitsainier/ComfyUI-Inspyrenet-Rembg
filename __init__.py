from .Inspyrenet_Rembg import InspyrenetRembgAdvanced, TextOverlayConfig, TextOverlayBatch

NODE_CLASS_MAPPINGS = {
    "InspyrenetRembgAdvanced": InspyrenetRembgAdvanced,
    "TextOverlayConfig": TextOverlayConfig,
    "TextOverlayBatch": TextOverlayBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InspyrenetRembgAdvanced": "Inspyrenet Rembg Advanced",
    "TextOverlayConfig": "Text Overlay Config",
    "TextOverlayBatch": "Text Overlay Batch"
}

__all__ = ['NODE_CLASS_MAPPINGS', "NODE_DISPLAY_NAME_MAPPINGS"]
