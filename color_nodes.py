class HexToColor:
    """Hex to Color"""

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hex_string": ("STRING", {"default": "#000000"}),
            },
        }

    CATEGORY = "utils/color"
    RETURN_TYPES = ("COLOR",)
    RETURN_NAMES = ("color",)

    FUNCTION = "execute"

    def execute(self, hex_string):
        # Ensure the hex string starts with #
        if not hex_string.startswith('#'):
            hex_string = '#' + hex_string
        return (hex_string,)

NODE_CLASS_MAPPINGS = {
    "Hex to Color": HexToColor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hex to Color": "Hex to Color",
}
