from PIL import Image, ImageOps, ImageSequence, ImageFile
import os
import numpy as np
import torch
import folder_paths
import node_helpers


def open_image_from_inputdir(filename):
    input_dir = folder_paths.get_input_directory()
    img_path = os.path.join(input_dir, filename)
    img = node_helpers.pillow(Image.open, img_path)
    img = node_helpers.pillow(ImageOps.exif_transpose, img)
    return img
    
def _process_layer(layer):
    img = open_image_from_inputdir(layer['image'])
    img = img.resize((layer['width'], layer['height']),Image.LANCZOS)
    img = img.rotate(layer['rotation'], expand=True)
    return img

def _paste_image(img, position,size):
    full_size_img = Image.new('RGBA',size, (0, 0, 0, 0))
    full_size_img.paste(img, position, img)
    return full_size_img
    
def fuse_layer(layerInfoArray):
    
    canvas = Image.new('RGBA', (layerInfoArray['width'], layerInfoArray['height']), (0, 0, 0, 0))
    type_product_layers = Image.new('RGBA', (layerInfoArray['width'], layerInfoArray['height']), (0, 0, 0, 0))
    
    for layer in layerInfoArray['layers']:
        img = _process_layer(layer)
        position = (layer['position_x'], layer['position_y'])
        
        full_size_img = _paste_image(img, position,canvas.size)
        
        canvas = Image.alpha_composite(canvas, full_size_img)
        if layer['type'] == 'Product':
            # 对于 type=Product 的图层，我们将其合并到 type_product_layers
            type_product_layers = Image.alpha_composite(type_product_layers, full_size_img)
        else: 
            # 从 type_product_layers 中减去当前图层遮盖的部分
            mask = full_size_img.split()[3]  # 使用当前图层的 alpha 通道作为蒙版
            inverted_mask = Image.eval(mask, lambda x: 255 - x)
            
            # 创建一个新的 Image 对象，包含 type_product_layers 中未被当前图层遮盖的部分
            unmasked_type_product = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
            unmasked_type_product.paste(type_product_layers, (0, 0), inverted_mask)
            
            # 更新 type_product_layers
            type_product_layers = unmasked_type_product
    return (canvas, type_product_layers,)
    
def pilimage_to_tensor(image,mask=False):
    if not isinstance(image, Image.Image):
        raise ValueError("`image` must be a PIL Image object.")
    # 转换为RGB并转为numpy数组
    image_array = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    
    # 转换为torch tensor
    image_tensor = torch.from_numpy(image_array).unsqueeze(0)
    
    if mask:
        mask = np.array(image.getchannel('A')).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)
        mask = 1. - mask
        mask = mask.unsqueeze(0)
        return image_tensor, mask
    else:
        return image_tensor