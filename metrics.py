import numpy as np
from skimage.io import imread
import math
import matplotlib.pyplot as plt
import csv
import os
import argparse


def calculate_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)


def calculate_psnr(image1, image2):
    mse = calculate_mse(image1, image2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    psnr = 10 * math.log10(max_pixel**2 / mse)
    return psnr


def main(wavelet_type, original_image_path, stego_image_path):
    original_image = imread(original_image_path)
    stego_image = imread(stego_image_path)

    # Convert images to the same number of channels if necessary
    if original_image.shape[2] == 4:
        original_image = original_image[:, :, :3]  # Discard alpha channel
    if stego_image.shape[2] == 4:
        stego_image = stego_image[:, :, :3]  # Discard alpha channel

    # Print the dimensions of the images before cropping
    print(f"Original image dimensions: {original_image.shape}")
    print(f"Stego image dimensions: {stego_image.shape}")

    # Ensure both images are of the same dimensions
    min_height = min(original_image.shape[0], stego_image.shape[0])
    min_width = min(original_image.shape[1], stego_image.shape[1])

    original_cropped = original_image[:min_height, :min_width]
    stego_cropped = stego_image[:min_height, :min_width]

    # Calculate MSE and PSNR
    mse_value = calculate_mse(original_cropped, stego_cropped)
    psnr_value = calculate_psnr(original_cropped, stego_cropped)
    print(f"MSE between the original and stego image: {mse_value}")
    print(f"PSNR between the original and stego image: {psnr_value} dB")

    # Save results to a CSV file
    csv_filename = "image_metrics.csv"
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                [
                    "Original Image",
                    "Stego Image",
                    "Original Image Dimensions",
                    "Stego Image Dimensions",
                    "Wavelet",
                    "MSE",
                    "PSNR",
                ]
            )
        writer.writerow(
            [
                original_image_path,
                stego_image_path,
                original_image.shape,
                stego_image.shape,
                wavelet_type,
                mse_value,
                psnr_value,
            ]
        )

    print(f"Results saved to {csv_filename}")

    # Display the images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_cropped, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(stego_cropped, cmap="gray")
    axes[1].set_title("Stego Image")
    axes[1].axis("off")

    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate stego images and save metrics to CSV."
    )
    parser.add_argument(
        "--wavelet_type", type=str, required=True, help="Type of wavelet used."
    )
    parser.add_argument(
        "--original_image", type=str, required=True, help="Path to the original image."
    )
    parser.add_argument(
        "--stego_image", type=str, required=True, help="Path to the stego image."
    )
    args = parser.parse_args()

    main(args.wavelet_type, args.original_image, args.stego_image)
