from .image_utils import _auto_analyze_parameters, hl_frequency_detail_restore, resize_image_with_padding,remove_alpha,add_alpha


class ImageFrequencyAnalyzer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "detail_image": ("IMAGE",),
                "mask_blur": ("INT", {"default": 16, "min": 0, "max": 1023}),
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("high_freq", "low_freq")
    FUNCTION = "analyzer_fc"

    CATEGORY = "utils"

    def analyzer_fc(self, image, detail_image, mask_blur, mask=None):
        high_freq, low_freq = _auto_analyze_parameters(image, detail_image, mask_blur, mask)
        print(f"Auto-analyzed parameters: keep_high_freq={high_freq}, erase_low_freq={low_freq}")
        return (high_freq, low_freq)


class HLFrequencyDetailRestore:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detail_image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "restore"
    CATEGORY = "utils"

    def restore(self, image, detail_image, mask=None):
        return (hl_frequency_detail_restore(image, detail_image, mask),)


class ImageResizeWithPadding:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 0, "tooltip": "Target width. Note that it will not crop the original image, but will scale proportionally based on width and height, then pad"}),
                "height": ("INT", {"default": 0, "tooltip": "Target height. Note that it will not crop the original image, but will scale proportionally based on width and height, then pad"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "reize"
    CATEGORY = "utils"

    def reize(self, image, width, height):
        return (resize_image_with_padding(image, width, height),)

class ImageRemoveAlpha:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "fill_color": ("COLOR", {"default": "#000000"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_alpha"
    CATEGORY = "utils"

    def remove_alpha(self, image, fill_color):
        return (remove_alpha(image, fill_color),)
    
class ImageAddAlpha:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_alpha"
    CATEGORY = "utils"

    def add_alpha(self, image, mask):
        return (add_alpha(image, mask),)


NODE_CLASS_MAPPINGS = {
    # "Image Frequency Analyzer": ImageFrequencyAnalyzer,
    # "HLFrequencyDetailRestore": HLFrequencyDetailRestore,
    "Image Resize With Padding": ImageResizeWithPadding,
    "Image Remove Alpha": ImageRemoveAlpha,
    "Image Add Alpha": ImageAddAlpha,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # "Image Frequency Analyzer": "Image Frequency Analyzer",
    # "HLFrequencyDetailRestore": "HLFrequencyDetailRestore",
    "Image Resize With Padding": "Image Resize With Padding",
    "Image Remove Alpha": "Image Remove Alpha",
    "Image Add Alpha": "Image Add Alpha",
}
