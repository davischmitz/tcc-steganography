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


# Based on Subhedar, Mansi S., and Vijay H. Mankar. "High capacity image steganography based on discrete wavelet transform and singular value decomposition."


def load_and_convert_to_rgb(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def load_grayscale(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


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


# Function to perform SVD
def apply_svd(matrix):
    U, S, V = np.linalg.svd(matrix, full_matrices=False)
    return U, S, V


# Embedding function with wavelet_type parameter
def embed_image(cover_image, secret_image, wavelet_type):
    # Perform one level DWT on the cover image
    coeffs_cover = pywt.dwt2(cover_image, wavelet_type)
    LL_cover, (LH_cover, HL_cover, HH_cover) = coeffs_cover

    # Perform one level DWT on the secret image
    secret_image = cv2.resize(
        secret_image, (cover_image.shape[1], cover_image.shape[0])
    )
    _, (_, _, HH_secret) = pywt.dwt2(secret_image, wavelet_type)

    # Apply SVD on the HH bands of both cover and secret images
    Uc, Sc, Vc = apply_svd(HH_cover)
    Us, Ss, Vs = apply_svd(HH_secret)

    # Replace the singular values of HH band of the cover with those from the secret image
    Sc_embed = Sc.copy()
    Sc_embed[: len(Ss)] = Ss  # Embed secret singular values

    # Reconstruct the HH band with modified singular values
    HH_embedded = np.dot(Uc, np.dot(np.diag(Sc_embed), Vc))

    # Reconstruct the stego image by performing inverse DWT
    coeffs_stego = (LL_cover, (LH_cover, HL_cover, HH_embedded))
    stego_image = pywt.idwt2(coeffs_stego, wavelet_type)
    return np.uint8(stego_image)


# Extraction function with wavelet_type parameter
def extract_image(stego_image, cover_image, wavelet_type):
    # Perform DWT on stego image and cover image
    _, (_, _, HH_stego) = pywt.dwt2(stego_image, wavelet_type)
    _, (_, _, HH_cover) = pywt.dwt2(cover_image, wavelet_type)

    # Apply SVD on HH bands of both stego and cover images
    Uc, Sc, Vc = apply_svd(HH_cover)
    Us, Ss, Vs = apply_svd(HH_stego)

    # Extract the singular values of the secret image
    S_extracted = Ss[: len(Sc)]  # Extract only necessary singular values

    # Reconstruct the HH band of the secret image using extracted singular values and original matrices
    HH_secret_reconstructed = np.dot(Us, np.dot(np.diag(S_extracted), Vs))

    # Reconstruct the secret image by inverse DWT
    coeffs_secret = (
        np.zeros_like(HH_secret_reconstructed),
        (
            np.zeros_like(HH_secret_reconstructed),
            np.zeros_like(HH_secret_reconstructed),
            HH_secret_reconstructed,
        ),
    )
    secret_image_reconstructed = pywt.idwt2(coeffs_secret, wavelet_type)
    return np.uint8(secret_image_reconstructed)


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
    cover_image = load_grayscale(cover_image_path)
    hidden_image = load_grayscale(hidden_image_path)

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

    stego_image = embed_image(cover_image, hidden_image, wavelet_type)

    extracted_hidden_image = extract_image(stego_image, cover_image, wavelet_type)

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