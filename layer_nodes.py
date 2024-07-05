import os
import json
import folder_paths
from .layer_utils import fuse_layer,pilimage_to_tensor,open_image_from_inputdir


class InitLayerInfoArray:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "width": ("INT", {"default": 0}),
                "height": ("INT", {"default": 0}),
                "prompt": ("STRING", {
                    "dynamicPrompts": False,
                    "multiline": True,
                    "default": ""
                }),
                "reference_bg": (sorted(files), {"image_upload": False}),
                "number_of_results": ("INT", {"default": 0}),
                "correct_color":("BOOLEAN", {"default": True,"lable_on":"active","lable_off":"inactive"}),
                "render_strength": (["Default", "Extra Weak", "Weak", "Strong", "Extra Stong"],),
                "color_strength": (["Default", "None", "Weak", "Strong", "Extra Stong"],),
                "outline_strength": (["Default", "None","Extra Weak", "Weak", "Strong", "Extra Stong"],),
            }
        }
    RETURN_TYPES = ("LAYER_INFO_ARRAY","STRING",)
    RETURN_NAMES = ("layerInfoArray","layerInfoArrayJson",)
    FUNCTION = "init_fc"

    CATEGORY = "utils"

    def init_fc(self,width,height,prompt,reference_bg,number_of_results,correct_color,render_strength,color_strength,outline_strength,):
        #将上面的内容转换成json对象
        layerInfoArray = {
            "width": width,
            "height": height,
            "prompt": prompt,
            "reference_bg": reference_bg,
            "number_of_results": number_of_results,
            "correct_color": correct_color,
            "render_strength": render_strength,
            "color_strength": color_strength,
            "outline_strength": outline_strength,
            "layers": []
        }
        return (layerInfoArray, json.dumps(layerInfoArray),)

class AddedLayerInfoToArray:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "layerInfoArray":('LAYER_INFO_ARRAY', {}),
                "type":  (["Product", "Prop", "Human-Hand", "Human-Male", "Human-Female"],),
                "image": (sorted(files), {"image_upload": False}),
                "width": ("INT", {"default": 0}),
                "height": ("INT", {"default": 0}),
                "rotation": ("INT", {"default": 0,"min":0,"max":360,"step":1}),
                "position_x": ("INT", {"default": 0}),
                "position_y": ("INT", {"default": 0}),
            }
        }
    RETURN_TYPES = ("LAYER_INFO_ARRAY","STRING",)
    RETURN_NAMES = ("layerInfoArray","layerInfoArrayJson",)
    FUNCTION = "add_fc"

    CATEGORY = "utils"

    def add_fc(self,layerInfoArray,type,image,width,height,rotation,position_x,position_y):
        layer = {
            "type": type,
            "image":image,
            "width": width,
            "height": height,
            "rotation": rotation,
            "position_x": position_x,
            "position_y": position_y,
        }
        layerInfoArray = json.loads(json.dumps(layerInfoArray))
        
        layerInfoArray["layers"].append(layer)
        return (layerInfoArray, json.dumps(layerInfoArray),)
    
    
class LayerInfoArrayFilter:
    @classmethod
    def INPUT_TYPES(s):
        return {}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "load_unet"

    CATEGORY = "x"

    def load_unet(self,):
        return ("unet",)
    
    
class LayerInfoArrayFuse:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "layerInfoArrayJson":("STRING",{"default": ""}),
            }
        }
    RETURN_TYPES = ("IMAGE","MASK","IMAGE","MASK","STRING","IMAGE","BOOLEAN","INT","INT","INT")
    RETURN_NAMES = ("fuseImage","fuseMask","fuseProductImage","fuseProductMask","prompt","referenceBg","correctColor","renderStrength","colorStrength","outlineStrength")
    FUNCTION = "fuse_fc"

    CATEGORY = "utils"

    def fuse_fc(self, layerInfoArrayJson,):
        layerInfoArray = json.loads(layerInfoArrayJson)

        fuse_image, fuse_product_image = fuse_layer(layerInfoArray)
        fuse_image, fuse_mask = pilimage_to_tensor(fuse_image, mask=True)
        fuse_product_image, fuse_product_mask = pilimage_to_tensor(
            fuse_product_image, mask=True)
        if layerInfoArray["reference_bg"] is not None and layerInfoArray["reference_bg"] != "":
            reference_bg_image = open_image_from_inputdir(
                layerInfoArray["reference_bg"])
            reference_bg_image = pilimage_to_tensor(reference_bg_image)
        else:
            reference_bg_image = None
        return (fuse_image,fuse_mask, fuse_product_image,fuse_product_mask, layerInfoArray["prompt"], reference_bg_image, layerInfoArray["correct_color"], 0, 0, 0,)
    
class LayerInfoArrayIPAdapterAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "load_unet"

    CATEGORY = "x"

    def load_unet(self,):
        return ("unet",)


NODE_CLASS_MAPPINGS = {
    "Init Layer Info Array":InitLayerInfoArray,
    "Added Layer Info To Array":AddedLayerInfoToArray,
    # "Layer Info Array Filter":LayerInfoArrayFilter,
    "Layer Info Array Fuse":LayerInfoArrayFuse,
    # "Layer Info Array IPAdapter Advanced":LayerInfoArrayIPAdapterAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Init Layer Info Array":"Init Layer Info Array",
    "Added Layer Info To Array": "Added Layer Info To Array",
    # "Layer Info Array Filter": "Layer Info Array Filter",
    "Layer Info Array Fuse": "Layer Info Array Fuse",
    # "Layer Info Array IPAdapter Advanced": "Layer Info Array IPAdapter Advanced",

}