import numpy as np
import torch
from scipy import fftpack
from skimage import color, exposure
from PIL import Image

def _tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def _auto_analyze_parameters(image, detail_image, mask_blur, mask=None):
    original = np.array(_tensor2pil(image[0]).convert('L'))
    redrawn = np.array(_tensor2pil(detail_image[0]).convert('L'))

    if mask is not None:
        mask = np.array(_tensor2pil(mask[0]).convert('L')) / 255.0
    else:
        mask = np.ones_like(original)

    keep_high_freq, erase_low_freq = _analyze_frequency(original, redrawn, mask)

    return keep_high_freq, erase_low_freq

def _analyze_frequency(original, redrawn, mask):
    # 计算傅里叶变换
    fft_original = fftpack.fft2(original * mask)
    fft_redrawn = fftpack.fft2(redrawn * mask)

    # 计算幅度谱
    magnitude_original = np.abs(fft_original)
    magnitude_redrawn = np.abs(fft_redrawn)

    # 计算差异
    diff = np.abs(magnitude_redrawn - magnitude_original)

    # 创建频率掩码
    rows, cols = original.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    distances = np.sqrt((y - crow)**2 + (x - ccol)**2)
    
    # 创建一个连续的频率权重掩码
    freq_weight = 1 / (1 + distances / (min(crow, ccol) / 4))

    # 计算加权差异
    weighted_diff = diff * freq_weight

    # 计算总体差异和加权差异
    total_diff = np.sum(diff)
    total_weighted_diff = np.sum(weighted_diff)

    # 计算相对差异
    relative_diff = total_weighted_diff / total_diff if total_diff > 0 else 0

    # 分析原始图像和重绘图像的亮度差异
    brightness_original = np.mean(original)
    brightness_redrawn = np.mean(redrawn)
    brightness_diff = brightness_redrawn - brightness_original

    # 根据亮度差异调整参数
    brightness_factor = 1 + brightness_diff / 255  # 归一化亮度差异

    # 计算 keep_high_freq
    keep_high_freq = int(1023 * (1 - np.exp(-5 * relative_diff * brightness_factor)))
    
    # 计算 erase_low_freq
    erase_low_freq = int(512 * (1 - np.exp(-3 * relative_diff / brightness_factor)))

    # 确保 keep_high_freq 和 erase_low_freq 在合理范围内
    keep_high_freq = max(min(keep_high_freq, 1023), 512)
    erase_low_freq = max(min(erase_low_freq, 512), 100)

    return keep_high_freq, erase_low_freq

# ... [之后的代码保持不变] ...

import numpy as np
import torch
from PIL import Image
import pywt
import cv2
from skimage.metrics import structural_similarity as ssim

def tensor_to_numpy(tensor):
    return np.clip(255.0 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

def numpy_to_tensor(array):
    return torch.from_numpy(array).float().div(255.0).unsqueeze(0)

def pad_image(img, pad=16):
    if len(img.shape) == 2:
        return np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')
    else:
        return np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

def unpad_image(img, pad=16):
    if len(img.shape) == 2:
        return img[pad:-pad, pad:-pad]
    else:
        return img[pad:-pad, pad:-pad, :]

def wavelet_decomposition(image, levels=3):
    coeffs = pywt.wavedec2(image, 'db4', level=levels)
    return coeffs

def wavelet_reconstruction(coeffs):
    return pywt.waverec2(coeffs, 'db4')

def blend_coefficients(orig_coeffs, redrawn_coeffs, mask):
    blended_coeffs = []
    for i, (orig_c, redrawn_c) in enumerate(zip(orig_coeffs, redrawn_coeffs)):
        if i == 0:  # Approximation coefficient
            resized_mask = cv2.resize(mask, (orig_c.shape[1], orig_c.shape[0]), interpolation=cv2.INTER_LINEAR)
            blended_c = orig_c * (1 - resized_mask) + redrawn_c * resized_mask
        else:  # Detail coefficients
            resized_mask = cv2.resize(mask, (orig_c[0].shape[1], orig_c[0].shape[0]), interpolation=cv2.INTER_LINEAR)
            ssim_map = np.array([ssim(orig_d, redrawn_d, data_range=redrawn_d.max() - redrawn_d.min(), full=True)[1] for orig_d, redrawn_d in zip(orig_c, redrawn_c)])
            blend_factor = resized_mask * (1 - ssim_map)
            blended_c = tuple(orig_d * (1 - bf) + redrawn_d * bf for orig_d, redrawn_d, bf in zip(orig_c, redrawn_c, blend_factor))
        blended_coeffs.append(blended_c)
    return blended_coeffs

def hl_frequency_detail_restore(image, detail_image, mask=None, mask_blur=0):
    original = tensor_to_numpy(image[0])
    redrawn = tensor_to_numpy(detail_image[0])
    
    if mask is not None:
        mask = tensor_to_numpy(mask[0]) / 255.0
        if len(mask.shape) == 3 and mask.shape[2] > 1:
            mask = mask[:,:,0]  # 使用第一个通道
        if mask_blur > 0:
            mask = cv2.GaussianBlur(mask, (mask_blur * 2 + 1, mask_blur * 2 + 1), 0)
    else:
        mask = np.ones((original.shape[0], original.shape[1]), dtype=np.float32)
    
    # 边缘扩展
    pad = 16
    original_padded = pad_image(original, pad)
    redrawn_padded = pad_image(redrawn, pad)
    mask_padded = pad_image(mask, pad)
    
    result = np.zeros_like(original_padded)
    
    for channel in range(3):
        orig_coeffs = wavelet_decomposition(original_padded[:,:,channel], levels=4)
        redrawn_coeffs = wavelet_decomposition(redrawn_padded[:,:,channel], levels=4)
        
        blended_coeffs = blend_coefficients(orig_coeffs, redrawn_coeffs, mask_padded)
        
        result[:,:,channel] = wavelet_reconstruction(blended_coeffs)
    
    # 裁剪回原始大小
    result = unpad_image(result, pad)
    
    # 使用mask进行alpha混合
    result = original * (1 - mask[:,:,np.newaxis]) + result * mask[:,:,np.newaxis]
    
    # 确保结果在0-255范围内
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # 局部对比度增强
    try:
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    except cv2.error as e:
        print(f"Warning: Error during color space conversion or CLAHE. Skipping this step. Error: {e}")
        # 如果发生错误，我们将使用原始的result
    
    return numpy_to_tensor(result)

def resize_image_with_padding(image,width,height):
    original = tensor_to_numpy(image[0])
    #等比缩放
    scale = min(width/original.shape[1],height/original.shape[0])
    new_size = (int(original.shape[1] * scale), int(original.shape[0] * scale))
    resized_image = cv2.resize(original, new_size, interpolation=cv2.INTER_LANCZOS4)
    # 判断宽高是否需要填充，如果需要进行左右填充
    if new_size[0] < width:
        offset = width - new_size[0]
        right_padding = offset // 2
        left_padding = offset - right_padding
        resized_image = np.pad(resized_image, ((0, 0), (left_padding, right_padding), (0, 0)), mode='constant')
        
    if new_size[1] < height:
        offset = height - new_size[1]
        bottom_padding = offset // 2
        top_padding = offset - bottom_padding
        resized_image = np.pad(resized_image, ((top_padding, bottom_padding), (0, 0), (0, 0)), mode='constant')
        
    return numpy_to_tensor(resized_image)
    
    