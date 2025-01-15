from .Inspyrenet_Rembg import InspyrenetRembgAdvanced

NODE_CLASS_MAPPINGS = {
    "InspyrenetRembgAdvanced": InspyrenetRembgAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InspyrenetRembgAdvanced": "Inspyrenet Rembg Advanced"
}
__all__ = ['NODE_CLASS_MAPPINGS', "NODE_DISPLAY_NAME_MAPPINGS"]
