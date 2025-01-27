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


def apply_svd(matrix):
    """Perform SVD on the given matrix."""
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    return u, s, vt


def embed_svd(cover_s, hidden_s, scale):
    """Embed hidden singular values into cover singular values with scaling."""
    return cover_s + scale * hidden_s


def embed_matrix(cover_band, hidden_matrix, scale):
    """Embed hidden matrix into the cover's detail band with scaling."""
    cover_u, cover_s, cover_vt = apply_svd(cover_band)
    embedded_s = cover_s + scale * np.diag(hidden_matrix)
    return np.dot(cover_u, np.dot(np.diag(embedded_s), cover_vt))


def extract_matrix(stego_band, scale):
    """Extract hidden matrix from the embedded detail band."""
    stego_u, stego_s, stego_vt = apply_svd(stego_band)
    extracted_matrix = (np.diag(stego_s) / scale).astype(
        np.float32
    )  # Scale down to retrieve hidden
    return extracted_matrix


def extract_hidden_svd(stego_s, scale):
    """Extract hidden singular values from the embedded singular values."""
    return stego_s / scale


def multi_level_dwt(image, wavelet, levels):
    """Perform multi-level DWT decomposition on an image."""
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=levels)
    return coeffs


def multi_level_idwt(coeffs, wavelet):
    """Perform multi-level inverse DWT to reconstruct an image."""
    return pywt.waverec2(coeffs, wavelet=wavelet)


def embed_in_highest_level_only_details(cover_dwt, hidden_dwt, scale):
    """Embed the hidden image into the high-frequency details of the highest DWT level."""
    cover_details = cover_dwt[1]  # High-frequency details at the highest level
    hidden_details = hidden_dwt[1]  # High-frequency details of hidden image

    embedded_details = []
    for cover_band, hidden_band in zip(cover_details, hidden_details):
        # Apply SVD on each band (horizontal, vertical, diagonal) of cover and hidden details
        cover_u, cover_s, cover_vt = apply_svd(cover_band)
        hidden_u, hidden_s, hidden_vt = apply_svd(hidden_band)

        # Embed hidden image's singular values into the cover's singular values
        embedded_s = embed_svd(cover_s, hidden_s, scale)

        # Reconstruct the embedded high-frequency band
        embedded_band = np.dot(cover_u, np.dot(np.diag(embedded_s), cover_vt))
        embedded_details.append(embedded_band)

    cover_dwt[1] = embedded_details
    return cover_dwt


## TODO works but requires the hidden DWT
def extract_from_highest_level_only_details(stego_dwt, hidden_dwt, scale):
    """
    Extract the hidden image from the high-frequency details of the highest DWT level.
    Assumes access to the original hidden DWT's U and Vt matrices and the stego DWT.

    Parameters:
    - stego_dwt: DWT coefficients of the stego image
    - hidden_dwt: DWT coefficients of the original hidden image
    - scale: The embedding scale factor used during encoding

    Returns:
    - hidden_dwt_reconstructed: DWT coefficients for the reconstructed hidden image
    """
    stego_details = stego_dwt[
        1
    ]  # High-frequency details at the highest level of stego image
    hidden_details = hidden_dwt[
        1
    ]  # High-frequency details at the highest level of hidden image

    extracted_details = []
    for stego_band, hidden_band in zip(stego_details, hidden_details):
        # Apply SVD on both the stego band and the hidden band
        stego_u, stego_s, stego_vt = apply_svd(stego_band)
        hidden_u, hidden_s, hidden_vt = apply_svd(hidden_band)

        # Extract the hidden singular values
        extracted_hidden_s = stego_s / scale

        # Reconstruct the hidden image's high-frequency band using hidden U and Vt
        hidden_band_reconstructed = np.dot(
            hidden_u, np.dot(np.diag(extracted_hidden_s), hidden_vt)
        )

        # Append the reconstructed high-frequency band to the extracted details
        extracted_details.append(hidden_band_reconstructed)

    # Replace the highest level's high-frequency details with the extracted hidden details
    hidden_dwt_reconstructed = [hidden_dwt[0], extracted_details]
    return hidden_dwt_reconstructed


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
            axes[1, 0].set_title("Aproximation (LL)")
            axes[1, 0].set_axis_off()
            continue
        # First row: Wavelet decomposition structure
        draw_2d_wp_basis(
            shape, wavedec2_keys(level), ax=axes[0, level], label_levels=label_levels
        )
        axes[0, level].set_title(f"Coefficients (Level {level})")
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
        plt.title(f"Histograma do Canal {color}")
        plt.xlabel("Intensidade dos Pixels")
        plt.ylabel("Densidade")
        plt.legend()

    plt.tight_layout()
    plt.show()


def plot_histograms_with_matplotlib(cover_image, stego_image):

    # Ensure values are within 0 to 255 range
    cover_image = np.clip(cover_image, 0, 255).astype(np.uint8)
    stego_image = np.clip(stego_image, 0, 255).astype(np.uint8)

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
        plt.title(f"Histograma do Canal {channel_names[i]}")
        plt.xlabel("Intensidade dos Pixels")
        plt.ylabel("Densidade")
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
    # Resize image2 to match the shape of image1 only if shapes differ
    if image1.shape != image2.shape:
        image2 = resize(image2, image1.shape, anti_aliasing=True)
    # Calculate MSE between image1 and (possibly resized) image2
    return np.mean((image1 - image2) ** 2)


def calculate_psnr(image1, image2):
    mse = calculate_mse(image1, image2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    psnr = 10 * math.log10(max_pixel**2 / mse)
    return psnr


def main(
    wavelet_type,
    embed_scale,
    cover_image_path,
    hidden_image_path,
    stego_image_path,
    encoding_dwt_level,
    decoding_dwt_level,
):
    # Configuration
    rows, cols = 2, 2
    print(f"Encoding DWT level: {encoding_dwt_level}")
    print(f"Decoding DWT level: {decoding_dwt_level}")

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

    # # Separate RGB channels
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
    cover_dwt_red = multi_level_dwt(cover_red, wavelet_type, encoding_dwt_level)
    cover_dwt_green = multi_level_dwt(cover_green, wavelet_type, encoding_dwt_level)
    cover_dwt_blue = multi_level_dwt(cover_blue, wavelet_type, encoding_dwt_level)

    hidden_dwt_red = multi_level_dwt(hidden_red, wavelet_type, encoding_dwt_level)
    hidden_dwt_green = multi_level_dwt(hidden_green, wavelet_type, encoding_dwt_level)
    hidden_dwt_blue = multi_level_dwt(hidden_blue, wavelet_type, encoding_dwt_level)

    # Display the coefficients
    display_multi_level_dwt_coefficients(
        cover_dwt_red, encoding_dwt_level, encoding_dwt_level
    )
    display_multi_level_dwt_coefficients(
        hidden_dwt_red, encoding_dwt_level, encoding_dwt_level
    )

    # Embed hidden image into highest-level details, storing U matrices
    stego_dwt_red = embed_in_highest_level_only_details(
        cover_dwt_red, hidden_dwt_red, embed_scale
    )
    stego_dwt_green = embed_in_highest_level_only_details(
        cover_dwt_green, hidden_dwt_green, embed_scale
    )
    stego_dwt_blue = embed_in_highest_level_only_details(
        cover_dwt_blue, hidden_dwt_blue, embed_scale
    )

    # Perform Inverse DWT to get the stego image
    stego_red = multi_level_idwt(stego_dwt_red, wavelet_type)
    stego_green = multi_level_idwt(stego_dwt_green, wavelet_type)
    stego_blue = multi_level_idwt(stego_dwt_blue, wavelet_type)

    # Merge stego channels to form final stego image
    stego_image = cv2.merge(
        (stego_red.astype(int), stego_green.astype(int), stego_blue.astype(int))
    )

    ####################################################
    # DECODING PROCESS
    ####################################################

    # Separate RGB channels
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

    # Decompose each channel to multiple DWT levels
    stego_dwt_red = multi_level_dwt(stego_red, wavelet_type, decoding_dwt_level)
    stego_dwt_green = multi_level_dwt(stego_green, wavelet_type, decoding_dwt_level)
    stego_dwt_blue = multi_level_dwt(stego_blue, wavelet_type, decoding_dwt_level)

    cover_dwt_red = multi_level_dwt(cover_red, wavelet_type, decoding_dwt_level)
    cover_dwt_green = multi_level_dwt(cover_green, wavelet_type, decoding_dwt_level)
    cover_dwt_blue = multi_level_dwt(cover_blue, wavelet_type, decoding_dwt_level)

    # Extract hidden DWT coefficients for each RGB channel
    extracted_hidden_dwt_red = extract_from_highest_level_only_details(
        stego_dwt_red, hidden_dwt_red, embed_scale
    )
    extracted_hidden_dwt_green = extract_from_highest_level_only_details(
        stego_dwt_green, hidden_dwt_green, embed_scale
    )
    extracted_hidden_dwt_blue = extract_from_highest_level_only_details(
        stego_dwt_blue, hidden_dwt_blue, embed_scale
    )

    # Perform Inverse DWT to get the hidden image
    extracted_hidden_red = multi_level_idwt(extracted_hidden_dwt_red, wavelet_type)
    extracted_hidden_green = multi_level_idwt(extracted_hidden_dwt_green, wavelet_type)
    extracted_hidden_blue = multi_level_idwt(extracted_hidden_dwt_blue, wavelet_type)

    # Merge RGB channels to form the final extracted hidden image
    extracted_hidden_image = cv2.merge(
        (
            extracted_hidden_red.astype(int),
            extracted_hidden_green.astype(int),
            extracted_hidden_blue.astype(int),
        )
    )

    # Display cover and hidden images
    display_image_subplot(cover_image, 1, "Cover Image", rows, cols)
    display_image_subplot(hidden_image, 2, "Embedded Image", rows, cols)
    display_image_subplot(stego_image, 3, "Stego Image", rows, cols)
    display_image_subplot(
        extracted_hidden_image, 4, "Extracted Embedded Image", rows, cols
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
    psnr_value_hidden_extracted = calculate_psnr(hidden_image, extracted_hidden_image)
    print(f"MSE between the original and stego image: {mse_value}")
    print(f"PSNR between the original and stego image: {psnr_value} dB")
    print(
        f"PSNR between the hidden and extracted hidden image: {psnr_value_hidden_extracted} dB"
    )

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
                    "PSNR between cover / stego",
                    "PSNR between hidden / extracted",
                    "DWT Level Embedding",
                    "DWT Level Extraction",
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
                psnr_value_hidden_extracted,
                encoding_dwt_level,
                decoding_dwt_level,
            ]
        )

    print(f"Results saved to {csv_filename}")

        # Compare cover and stego images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cover_image, cmap="gray")
    axes[0].set_title("Cover Image")
    axes[0].axis("off")

    axes[1].imshow(stego_image, cmap="gray")
    axes[1].set_title("Stego Image")
    axes[1].axis("off")

    plt.show()


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
    parser.add_argument(
        "--embed_dwt_level",
        type=int,
        required=True,
        help="DWT level in which the embedding is performed.",
    )
    parser.add_argument(
        "--extract_dwt_level",
        type=int,
        required=True,
        help="DWT level in which the extraction is performed.",
    )
    args = parser.parse_args()

    main(
        args.wavelet_type,
        args.embed_scale,
        args.cover_image,
        args.hidden_image,
        args.stego_image,
        args.embed_dwt_level,
        args.extract_dwt_level,
    )
