import numpy as np
import pywt
from pywt._doc_utils import draw_2d_wp_basis, wavedec2_keys
from matplotlib import pyplot as plt
import cv2
import argparse
import math
import csv
import os
import seaborn as sns
import pandas as pd
from skimage.transform import resize


def load_and_convert_to_rgb(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def display_image_subplot(image, position, title, rows, cols):
    plt.subplot(rows, cols, position)
    plt.axis("off")
    plt.title(title)
    plt.imshow(image, aspect="equal")


def dwt_decompose(image, level=1):
    """Apply DWT decomposition to an image."""
    coeffs = pywt.wavedec2(image, "haar", level=level)
    return coeffs


def dwt_recompose(coeffs):
    """Recompose an image from its DWT coefficients."""
    return pywt.waverec2(coeffs, "haar")


def find_best_match(block, candidates):
    """Find the best-matching block using RMSE."""
    min_error = float("inf")
    best_match = None
    for candidate in candidates:
        error = np.sqrt(np.mean((block - candidate) ** 2))
        if error < min_error:
            min_error = error
            best_match = candidate
    return best_match, min_error


def embed_secret_in_cover(cover, secret, block_size=4):
    """Embed the secret image in the cover image and return the stego image along with the secret blocks."""
    # Step 1: Decompose cover and secret images using DWT
    cover_coeffs = dwt_decompose(cover)
    secret_coeffs = dwt_decompose(secret)

    cover_ll, (cover_hl, cover_lh, cover_hh) = cover_coeffs
    secret_ll, (secret_hl, secret_lh, secret_hh) = secret_coeffs

    # Step 2: Partition the LL sub-band of the secret and cover images into 4x4 blocks
    blocks_secret_ll = [
        secret_ll[i : i + block_size, j : j + block_size]
        for i in range(0, secret_ll.shape[0], block_size)
        for j in range(0, secret_ll.shape[1], block_size)
    ]
    blocks_cover_ll = [
        cover_ll[i : i + block_size, j : j + block_size]
        for i in range(0, cover_ll.shape[0], block_size)
        for j in range(0, cover_ll.shape[1], block_size)
    ]

    # Step 3: Embed each block from secret_ll into cover_hl via best match
    embedded_hl = np.copy(cover_hl)
    for idx, secret_block in enumerate(blocks_secret_ll):
        print(secret_block)
        # Find best match in cover's LL sub-band
        best_match, _ = find_best_match(secret_block, blocks_cover_ll)

        # Calculate the error block
        error_block = best_match - secret_block

        # Embed the error block into the HL sub-band of the cover
        i, j = (idx // (cover_hl.shape[1] // block_size)) * block_size, (
            idx % (cover_hl.shape[1] // block_size)
        ) * block_size
        embedded_hl[i : i + block_size, j : j + block_size] = error_block

    # Step 4: Reconstruct the stego image by performing inverse DWT
    stego_coeffs = (cover_ll, (embedded_hl, cover_lh, cover_hh))
    stego_image = dwt_recompose(stego_coeffs)

    # Return the stego image and the secret blocks (LL blocks of the secret image)
    return stego_image, blocks_secret_ll


def decode_secret_from_stego(stego_image, secret_blocks, block_size=4):
    """Decode the hidden image from the stego image using the secret blocks."""
    # Step 1: Decompose the stego image using DWT to extract the HL sub-band
    stego_coeffs = dwt_decompose(stego_image)
    stego_ll, (stego_hl, _, _) = stego_coeffs

    # Step 2: Partition the stego's HL sub-band into 4x4 blocks
    blocks_stego_hl = [
        stego_hl[i : i + block_size, j : j + block_size]
        for i in range(0, stego_hl.shape[0], block_size)
        for j in range(0, stego_hl.shape[1], block_size)
    ]

    # Step 3: Decode the secret image's blocks from the stego's HL sub-band
    reconstructed_secret_ll = np.zeros_like(stego_ll)

    for idx, error_block in enumerate(blocks_stego_hl):
        # Get the corresponding secret block
        secret_block = secret_blocks[idx]

        # Recover the original cover block by reversing the error embedding
        cover_block = secret_block + error_block

        # Place the recovered block in the reconstructed secret image
        i, j = (idx // (stego_hl.shape[1] // block_size)) * block_size, (
            idx % (stego_hl.shape[1] // block_size)
        ) * block_size
        reconstructed_secret_ll[i : i + block_size, j : j + block_size] = cover_block

    # Step 4: Reconstruct the full secret image from the decoded LL band
    decoded_secret_coeffs = (
        reconstructed_secret_ll,
        (np.zeros_like(stego_hl), np.zeros_like(stego_hl), np.zeros_like(stego_hl)),
    )
    decoded_secret = dwt_recompose(decoded_secret_coeffs)
    return decoded_secret


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


def plot_histograms(cover_image, stego_image):
    # Flatten the RGB channels for both images
    cover_red, cover_green, cover_blue = (
        cover_image[:, :, 0].flatten(),
        cover_image[:, :, 1].flatten(),
        cover_image[:, :, 2].flatten(),
    )
    stego_red, stego_green, stego_blue = (
        stego_image[:, :, 0].flatten(),
        stego_image[:, :, 1].flatten(),
        stego_image[:, :, 2].flatten(),
    )

    # Create a DataFrame for plotting with Seaborn
    data = pd.DataFrame(
        {
            "Cover Red": cover_red,
            "Cover Green": cover_green,
            "Cover Blue": cover_blue,
            "Stego Red": stego_red,
            "Stego Green": stego_green,
            "Stego Blue": stego_blue,
        }
    )

    # Plot histograms
    plt.figure(figsize=(18, 8))
    channels = ["Red", "Green", "Blue"]
    for i, color in enumerate(channels):
        plt.subplot(1, 3, i + 1)
        sns.histplot(
            data[f"Cover {color}"],
            color=color.lower(),
            label="Cover",
            kde=True,
            stat="density",
            element="step",
            linewidth=1.5,
        )
        sns.histplot(
            data[f"Stego {color}"],
            color=color.lower(),
            label="Stego",
            kde=True,
            stat="density",
            element="step",
            linestyle="--",
            linewidth=1.5,
        )
        plt.title(f"{color} Channel Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Density")
        plt.legend()

    plt.tight_layout()
    plt.show()


def plot_histograms_with_matplotlib(cover_image, stego_image):
    # Flatten the RGB channels for both images
    cover_red, cover_green, cover_blue = (
        cover_image[:, :, 0].flatten(),
        cover_image[:, :, 1].flatten(),
        cover_image[:, :, 2].flatten(),
    )
    stego_red, stego_green, stego_blue = (
        stego_image[:, :, 0].flatten(),
        stego_image[:, :, 1].flatten(),
        stego_image[:, :, 2].flatten(),
    )

    # Define colors for the channels
    colors = ["red", "green", "blue"]
    cover_channels = [cover_red, cover_green, cover_blue]
    stego_channels = [stego_red, stego_green, stego_blue]
    channel_names = ["Red", "Green", "Blue"]

    # Plot histograms for each channel
    plt.figure(figsize=(18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.hist(
            cover_channels[i],
            bins=50,
            color=colors[i],
            alpha=0.5,
            label="Cover",
            density=True,
        )
        plt.hist(
            stego_channels[i],
            bins=50,
            color=colors[i],
            alpha=0.3,
            label="Stego",
            density=True,
            linestyle="--",
        )
        plt.title(f"{channel_names[i]} Channel Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Density")
        plt.legend()

    plt.tight_layout()
    plt.show()


def plot_combined_histogram_with_matplotlib(cover_image, stego_image):
    # Flatten the RGB channels across all channels (R, G, B) for both images
    cover_pixels = cover_image.flatten()
    stego_pixels = stego_image.flatten()

    # Plot combined histogram
    plt.figure(figsize=(10, 6))
    plt.hist(
        cover_pixels,
        bins=50,
        color="blue",
        alpha=0.5,
        label="Cover Image",
        density=True,
    )
    plt.hist(
        stego_pixels, bins=50, color="red", alpha=0.5, label="Stego Image", density=True
    )
    plt.title("Combined Histogram of Cover and Stego Images")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Density")
    plt.legend()

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
    dwt_level = 1
    compression_ranks = 100

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

    stego_image, secret_blocks = embed_secret_in_cover(cover_image, hidden_image)

    plt.imshow(stego_image)
    plt.show()

    extracted_hidden_image = decode_secret_from_stego(stego_image, secret_blocks)

    # # # Separate RGB channels
    # cover_red, cover_green, cover_blue = (
    #     cover_image[:, :, 0],
    #     cover_image[:, :, 1],
    #     cover_image[:, :, 2],
    # )
    # hidden_red, hidden_green, hidden_blue = (
    #     hidden_image[:, :, 0],
    #     hidden_image[:, :, 1],
    #     hidden_image[:, :, 2],
    # )

    # # Apply multi-level DWT on RGB channels of both cover and hidden images
    # cover_dwt_red = multi_level_dwt(cover_red, wavelet_type, dwt_level)
    # cover_dwt_green = multi_level_dwt(cover_green, wavelet_type, dwt_level)
    # cover_dwt_blue = multi_level_dwt(cover_blue, wavelet_type, dwt_level)

    # hidden_dwt_red = multi_level_dwt(hidden_red, wavelet_type, dwt_level)
    # hidden_dwt_green = multi_level_dwt(hidden_green, wavelet_type, dwt_level)
    # hidden_dwt_blue = multi_level_dwt(hidden_blue, wavelet_type, dwt_level)

    # # Display the coefficients
    # display_multi_level_dwt_coefficients(cover_dwt_red, dwt_level, dwt_level)
    # display_multi_level_dwt_coefficients(hidden_dwt_red, dwt_level, dwt_level)

    ####################################################
    # DECODING PROCESS
    ####################################################

    # Display cover and hidden images
    display_image_subplot(cover_image, 1, "Cover Image", rows, cols)
    display_image_subplot(hidden_image, 2, "Image to Hide", rows, cols)
    display_image_subplot(stego_image, 3, "Stego Image", rows, cols)
    display_image_subplot(
        extracted_hidden_image, 4, "Extracted Hidden Image", rows, cols
    )
    plt.show()

    plot_histograms_with_matplotlib(cover_image, stego_image)

    # Save stego image
    save_image(
        stego_image,
        stego_image_path,
        size=(float(stego_image.shape[1]) / 100, float(stego_image.shape[0]) / 100),
    )

    # Save extracted hidden image
    # save_image(
    #     extracted_hidden_image,
    #     "extracted_hidden_image.tif",
    #     size=(
    #         float(extracted_hidden_image.shape[1]) / 100,
    #         float(extracted_hidden_image.shape[0]) / 100,
    #     ),
    # )

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
                "Multilevel Higher Band",
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
