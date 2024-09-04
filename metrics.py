import numpy as np
from skimage.io import imread
import math

def calculate_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

def calculate_psnr(image1, image2):
    mse = calculate_mse(image1, image2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

# Read the stego image
stego_image = imread("stego.png")

# Ensure both images are of the same dimensions
min_height = min(to_send_og.shape[0], stego_image.shape[0])
min_width = min(to_send_og.shape[1], stego_image.shape[1])

to_send_cropped = to_send_og[:min_height, :min_width]
stego_cropped = stego_image[:min_height, :min_width]

# Calculate PSNR
psnr_value = calculate_psnr(to_send_cropped, stego_cropped)
print(f"PSNR between the original and stego image: {psnr_value} dB")