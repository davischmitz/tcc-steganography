import numpy as np
import pywt
from pywt._doc_utils import draw_2d_wp_basis, wavedec2_keys
from matplotlib import pyplot as plt
import cv2
from skimage.transform import resize
import argparse
import math
import csv
import os


def load_and_convert_to_rgb(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def compute_multi_level_dwt(image, wavelet_type, level):
    return pywt.wavedec2(image, wavelet_type, level=level)


def compute_multi_level_idwt(coeffs, wavelet_type):
    return pywt.waverec2(coeffs, wavelet_type)


def modify_high_freq_bands(cover_coeffs, hidden_coeffs, embed_scale):
    approx_cover, details_cover = cover_coeffs[0], cover_coeffs[1:]
    approx_hidden, details_hidden = hidden_coeffs[0], hidden_coeffs[1:]

    modified_details = []
    for cover_detail, hidden_detail in zip(details_cover, details_hidden):
        modified_bands = []
        for cover_band, hidden_band in zip(cover_detail, hidden_detail):
            cover_svd = apply_svd(cover_band)
            hidden_svd = apply_svd(hidden_band)
            embedded_s = embed_svd(cover_svd, hidden_svd, embed_scale)
            modified_band = decode_svd(embedded_s, cover_svd[0], cover_svd[2])
            modified_bands.append(modified_band)
        modified_details.append(tuple(modified_bands))

    return [approx_cover] + modified_details


def extract_hidden_image(
    stego_image, cover_image, wavelet_type, embed_scale, dwt_level
):
    # Separate the RGB channels of both stego and cover images
    stego_red, stego_green, stego_blue = (
        stego_image[:, :, 0],
        stego_image[:, :, 1],
        stego_image[:, :, 2],
    )
    cover_red, cover_green, cover_blue = (
        cover_image[:, :, 0],
        cover_image[:, :, 1],
        cover_image[:, :, 2],
    )

    # Perform multi-level DWT on the cover and stego image channels
    stego_dwt_red = compute_multi_level_dwt(stego_red, wavelet_type, dwt_level)
    stego_dwt_green = compute_multi_level_dwt(stego_green, wavelet_type, dwt_level)
    stego_dwt_blue = compute_multi_level_dwt(stego_blue, wavelet_type, dwt_level)

    cover_dwt_red = compute_multi_level_dwt(cover_red, wavelet_type, dwt_level)
    cover_dwt_green = compute_multi_level_dwt(cover_green, wavelet_type, dwt_level)
    cover_dwt_blue = compute_multi_level_dwt(cover_blue, wavelet_type, dwt_level)

    # Extract hidden image high-frequency bands using inverse embedding logic
    hidden_dwt_red = extract_high_freq_bands(stego_dwt_red, cover_dwt_red, embed_scale)
    hidden_dwt_green = extract_high_freq_bands(
        stego_dwt_green, cover_dwt_green, embed_scale
    )
    hidden_dwt_blue = extract_high_freq_bands(
        stego_dwt_blue, cover_dwt_blue, embed_scale
    )

    # Perform Inverse DWT to get the hidden image from the extracted bands
    hidden_red = compute_multi_level_idwt(hidden_dwt_red, wavelet_type)
    hidden_green = compute_multi_level_idwt(hidden_dwt_green, wavelet_type)
    hidden_blue = compute_multi_level_idwt(hidden_dwt_blue, wavelet_type)

    # Merge hidden channels to form the extracted hidden image
    hidden_image = cv2.merge(
        (hidden_red.astype(int), hidden_green.astype(int), hidden_blue.astype(int))
    )

    return hidden_image


def extract_high_freq_bands(stego_coeffs, cover_coeffs, embed_scale):
    approx_cover, details_cover = cover_coeffs[0], cover_coeffs[1:]
    approx_stego, details_stego = stego_coeffs[0], stego_coeffs[1:]

    extracted_details = []
    for stego_detail, cover_detail in zip(details_stego, details_cover):
        extracted_bands = []
        for stego_band, cover_band in zip(stego_detail, cover_detail):
            # Apply SVD to both stego and cover bands
            cover_svd = apply_svd(cover_band)
            stego_svd = apply_svd(stego_band)
            # Extract hidden SVD values
            hidden_s = extract_hidden_svd(stego_svd[1], cover_svd[1], embed_scale)
            # Reconstruct the hidden detail band
            hidden_band = decode_svd(hidden_s, stego_svd[0], stego_svd[2])
            extracted_bands.append(hidden_band)
        extracted_details.append(tuple(extracted_bands))

    return [approx_cover] + extracted_details


def embed_svd(cover_svd, hidden_svd, scale):
    cover_u, cover_s, cover_vh = cover_svd
    hidden_u, hidden_s, hidden_vh = hidden_svd
    embedded_s = cover_s + scale * hidden_s
    return embedded_s


def apply_svd(matrix):
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    return u, singular_values, vh


def decode_svd(embedded_matrix, u, vh):
    return np.dot(u * embedded_matrix, vh)


def extract_hidden_svd(stego_s, cover_s, scale):
    hidden_s = (stego_s - cover_s) / scale
    return hidden_s


# def extract_hidden_svd(embedded_svd, scale):
#     return embedded_svd / scale


def display_image_subplot(image, position, title, rows, cols):
    plt.subplot(rows, cols, position)
    plt.axis("off")
    plt.title(title)
    plt.imshow(image, aspect="equal")


def display_multi_level_dwt_coefficients(coefficients, max_level, label_levels):
    shape = coefficients[0].shape
    fig, axes = plt.subplots(2, max_level + 1, figsize=[14, 8])

    for level in range(0, max_level + 1):
        if level == 0:
            # First column: Show the original approximation coefficients (the LL coefficients)
            axes[0, 0].set_axis_off()  # No wavelet diagram for original image
            axes[1, 0].imshow(coefficients[0], cmap=plt.cm.gray)
            axes[1, 0].set_title("Approximation (Level 0)")
            axes[1, 0].set_axis_off()
            continue

        # First row: Wavelet decomposition structure
        draw_2d_wp_basis(
            shape, wavedec2_keys(level), ax=axes[0, level], label_levels=label_levels
        )
        axes[0, level].set_title(f"{level} level decomposition")

        # Second row: Show the coefficients for the current level
        coeffs = coefficients[: level + 1]  # Slice coefficients up to current level
        arr, slices = pywt.coeffs_to_array(coeffs)
        axes[1, level].imshow(arr, cmap=plt.cm.gray)
        axes[1, level].set_title(f"Coefficients (Level {level})")
        axes[1, level].set_axis_off()

    plt.tight_layout()
    plt.show()


def save_image(image, file_path, size=None):
    fig = plt.figure(frameon=False)
    if size is not None:
        fig.set_size_inches(size[0], size[1])
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, aspect="auto")
    fig.savefig(file_path)

    ####################################################
    # METRICS
    ####################################################


def calculate_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)


def calculate_psnr(image1, image2):
    mse = calculate_mse(image1, image2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    psnr = 10 * math.log10(max_pixel**2 / mse)
    return psnr


def main(
    wavelet_type, embed_scale, cover_image_path, hidden_image_path, stego_image_path
):
    # Configuration
    rows, cols = 2, 2
    dwt_level = 4

    # Load images
    cover_image = load_and_convert_to_rgb(cover_image_path)
    hidden_image = load_and_convert_to_rgb(hidden_image_path)

    # Crop images to the same size
    print(f"Cover image dimensions: {cover_image.shape}")
    print(f"Hidden image dimensions: {hidden_image.shape}")
    min_height = min(cover_image.shape[0], hidden_image.shape[0])
    min_width = min(cover_image.shape[1], hidden_image.shape[1])
    cover_image = cover_image[:min_height, :min_width]
    hidden_image = hidden_image[:min_height, :min_width]

    ####################################################
    # ENCODING PROCESS
    ####################################################

    # Separate RGB channels
    cover_red, cover_green, cover_blue = (
        cover_image[:, :, 0],
        cover_image[:, :, 1],
        cover_image[:, :, 2],
    )
    hidden_red, hidden_green, hidden_blue = (
        hidden_image[:, :, 0],
        hidden_image[:, :, 1],
        hidden_image[:, :, 2],
    )

    # Apply multi-level DWT on RGB channels of both cover and hidden images
    cover_dwt_red = compute_multi_level_dwt(cover_red, wavelet_type, dwt_level)
    cover_dwt_green = compute_multi_level_dwt(cover_green, wavelet_type, dwt_level)
    cover_dwt_blue = compute_multi_level_dwt(cover_blue, wavelet_type, dwt_level)

    hidden_dwt_red = compute_multi_level_dwt(hidden_red, wavelet_type, dwt_level)
    hidden_dwt_green = compute_multi_level_dwt(hidden_green, wavelet_type, dwt_level)
    hidden_dwt_blue = compute_multi_level_dwt(hidden_blue, wavelet_type, dwt_level)

    # Display the coefficients
    display_multi_level_dwt_coefficients(cover_dwt_red, dwt_level, dwt_level)

    # Modify the high-frequency bands with the hidden image information
    modified_dwt_red = modify_high_freq_bands(
        cover_dwt_red, hidden_dwt_red, embed_scale
    )
    modified_dwt_green = modify_high_freq_bands(
        cover_dwt_green, hidden_dwt_green, embed_scale
    )
    modified_dwt_blue = modify_high_freq_bands(
        cover_dwt_blue, hidden_dwt_blue, embed_scale
    )

    # Perform Inverse DWT to get the stego image
    stego_red = compute_multi_level_idwt(modified_dwt_red, wavelet_type)
    stego_green = compute_multi_level_idwt(modified_dwt_green, wavelet_type)
    stego_blue = compute_multi_level_idwt(modified_dwt_blue, wavelet_type)

    # Merge stego channels to form final stego image
    stego_image = cv2.merge(
        (stego_red.astype(int), stego_green.astype(int), stego_blue.astype(int))
    )

    ####################################################
    # DECODING PROCESS
    ####################################################

    # Extract hidden image
    extracted_hidden_image = extract_hidden_image(
        stego_image, cover_image, wavelet_type, embed_scale, dwt_level
    )

    # Display cover and hidden images
    display_image_subplot(cover_image, 1, "Cover Image", rows, cols)
    display_image_subplot(hidden_image, 2, "Image to Hide", rows, cols)
    display_image_subplot(stego_image, 3, "Stego Image", rows, cols)
    display_image_subplot(hidden_image, 4, "Extracted Hidden Image", rows, cols)

    plt.show()

    # Save stego image
    save_image(
        stego_image,
        stego_image_path,
        size=(float(stego_image.shape[1]) / 100, float(stego_image.shape[0]) / 100),
    )

    # Save extracted hidden image
    save_image(
        extracted_hidden_image,
        "extracted_hidden_image.tif",
        size=(
            float(extracted_hidden_image.shape[1]) / 100,
            float(extracted_hidden_image.shape[0]) / 100,
        ),
    )

    print(f"Cover image dimensions: {cover_image.shape}")
    print(f"Hidden image dimensions: {hidden_image.shape}")
    print(f"Stego image dimensions: {stego_image.shape}")

    # Calculate MSE and PSNR
    mse_value = calculate_mse(cover_image, stego_image)
    psnr_value = calculate_psnr(cover_image, stego_image)
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
                    "Stego Strategy",
                    "Original Image",
                    "Stego Image",
                    "Original Image Dimensions",
                    "Stego Image Dimensions",
                    "Wavelet",
                    "Embedding Scale",
                    "MSE",
                    "PSNR",
                ]
            )
        writer.writerow(
            [
                "Higher Band",
                cover_image_path,
                hidden_image_path,
                cover_image.shape,
                stego_image.shape,
                wavelet_type,
                embed_scale,
                mse_value,
                psnr_value,
            ]
        )

    print(f"Results saved to {csv_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Steganography using higher band signals."
    )
    parser.add_argument(
        "--wavelet_type", type=str, required=True, help="Type of wavelet to use."
    )
    parser.add_argument(
        "--embed_scale", type=float, required=True, help="Embedding scale factor."
    )
    parser.add_argument(
        "--cover_image", type=str, required=True, help="Path to the cover image."
    )
    parser.add_argument(
        "--hidden_image", type=str, required=True, help="Path to the hidden image."
    )
    parser.add_argument(
        "--stego_image",
        type=str,
        required=True,
        help="Path to save the stego image.",
    )
    args = parser.parse_args()

    main(
        args.wavelet_type,
        args.embed_scale,
        args.cover_image,
        args.hidden_image,
        args.stego_image,
    )
