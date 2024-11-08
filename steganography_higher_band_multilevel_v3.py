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


def embed_in_highest_level_approximation(cover_dwt, hidden_dwt, scale):
    """
    Embed the hidden image's approximation coefficients into the high-frequency details of the highest DWT level
    of the cover image by distributing the approximation coefficients across the detail bands.

    Parameters:
    - cover_dwt: DWT coefficients of the cover image
    - hidden_dwt: DWT coefficients of the hidden image
    - scale: The embedding scale factor

    Returns:
    - modified_dwt: DWT coefficients of the stego image
    - hidden_u_matrices: List of U matrices from the hidden image's SVD (one per part of approximation coefficients)
    - hidden_vt_matrices: List of Vt matrices from the hidden image's SVD (one per part of approximation coefficients)
    """
    cover_details = cover_dwt[
        -1
    ]  # High-frequency details at the highest level of the cover image
    hidden_approximation = hidden_dwt[
        0
    ]  # Approximation coefficients of the hidden image

    # Split the hidden approximation coefficients into three parts
    split_hidden_approximation = np.array_split(hidden_approximation, 3, axis=0)

    embedded_details = []
    hidden_u_matrices = (
        []
    )  # Store U matrices for each part of the approximation coefficients
    hidden_vt_matrices = (
        []
    )  # Store Vt matrices for each part of the approximation coefficients

    # Loop over each part of the split hidden approximation and each cover detail band
    for cover_band, hidden_part in zip(cover_details, split_hidden_approximation):
        # Resize hidden_part to match cover_band's shape if they differ
        if hidden_part.shape != cover_band.shape:
            hidden_part = cv2.resize(
                hidden_part, (cover_band.shape[1], cover_band.shape[0])
            )

        # Apply SVD on the hidden part
        hidden_u, hidden_s, hidden_vt = apply_svd(hidden_part)

        # Store the hidden U and Vt matrices for decoding
        hidden_u_matrices.append(hidden_u)
        hidden_vt_matrices.append(hidden_vt)

        # Apply SVD on the cover band
        cover_u, cover_s, cover_vt = apply_svd(cover_band)

        # Embed hidden image's singular values into the cover's singular values
        embedded_s = cover_s + scale * hidden_s

        # Reconstruct the embedded high-frequency band
        embedded_band = np.dot(cover_u, np.dot(np.diag(embedded_s), cover_vt))
        embedded_details.append(embedded_band)

    # Replace the highest level's high-frequency details with the embedded details
    modified_dwt = cover_dwt[:-1] + [tuple(embedded_details)]

    return modified_dwt, hidden_u_matrices, hidden_vt_matrices


## TODO works but there's a big penalty on embedding the approximation coefficients
def extract_from_highest_level_approximation(
    stego_dwt, hidden_u_matrices, hidden_vt_matrices, scale, original_shape
):
    """
    Extract the hidden image's approximation coefficients from the high-frequency details of the highest DWT level
    of the stego image using the known U and Vt matrices for each part of the approximation coefficients.

    Parameters:
    - stego_dwt: DWT coefficients of the stego image
    - hidden_u_matrices: List of U matrices from the hidden image's SVD (one per part of approximation coefficients)
    - hidden_vt_matrices: List of Vt matrices from the hidden image's SVD (one per part of approximation coefficients)
    - scale: The embedding scale factor

    Returns:
    - hidden_dwt_reconstructed: DWT coefficients for the reconstructed hidden image
    """
    stego_details = stego_dwt[
        -1
    ]  # High-frequency details at the highest level of the stego image

    extracted_parts = []
    for stego_band, hidden_u, hidden_vt in zip(
        stego_details, hidden_u_matrices, hidden_vt_matrices
    ):
        # Apply SVD on the stego high-frequency band
        _, stego_s, _ = apply_svd(stego_band)

        # Extract the hidden singular values
        extracted_hidden_s = stego_s / scale

        # Reconstruct the hidden approximation part using hidden U and Vt
        hidden_part_reconstructed = np.dot(
            hidden_u, np.dot(np.diag(extracted_hidden_s), hidden_vt)
        )

        # Append the reconstructed part of the approximation coefficients
        extracted_parts.append(hidden_part_reconstructed)

    # Combine the extracted parts to reconstruct the hidden approximation coefficients
    hidden_approximation_reconstructed = np.vstack(extracted_parts)

    # Resize the reconstructed approximation coefficients to match the original shape
    if hidden_approximation_reconstructed.shape != original_shape:
        hidden_approximation_reconstructed = cv2.resize(
            hidden_approximation_reconstructed, (original_shape[1], original_shape[0])
        )

    # Replace the hidden image's approximation coefficients in the DWT structure
    hidden_dwt_reconstructed = [hidden_approximation_reconstructed] + list(
        stego_dwt[1:]
    )
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
    dwt_level = 3

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
    cover_dwt_red = multi_level_dwt(cover_red, wavelet_type, dwt_level)
    cover_dwt_green = multi_level_dwt(cover_green, wavelet_type, dwt_level)
    cover_dwt_blue = multi_level_dwt(cover_blue, wavelet_type, dwt_level)

    hidden_dwt_red = multi_level_dwt(hidden_red, wavelet_type, dwt_level)
    hidden_dwt_green = multi_level_dwt(hidden_green, wavelet_type, dwt_level)
    hidden_dwt_blue = multi_level_dwt(hidden_blue, wavelet_type, dwt_level)

    # Display the coefficients
    display_multi_level_dwt_coefficients(cover_dwt_red, dwt_level, dwt_level)

    # Embed hidden image into highest-level details, storing U matrices
    stego_dwt_red, hidden_u_matrices_red, hidden_vt_matrices_red = (
        embed_in_highest_level_approximation(cover_dwt_red, hidden_dwt_red, embed_scale)
    )
    stego_dwt_green, hidden_u_matrices_green, hidden_vt_matrices_green = (
        embed_in_highest_level_approximation(
            cover_dwt_green, hidden_dwt_green, embed_scale
        )
    )
    stego_dwt_blue, hidden_u_matrices_blue, hidden_vt_matrices_blue = (
        embed_in_highest_level_approximation(
            cover_dwt_blue, hidden_dwt_blue, embed_scale
        )
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
    stego_dwt_red = multi_level_dwt(stego_red, wavelet_type, dwt_level)
    stego_dwt_green = multi_level_dwt(stego_green, wavelet_type, dwt_level)
    stego_dwt_blue = multi_level_dwt(stego_blue, wavelet_type, dwt_level)

    cover_dwt_red = multi_level_dwt(cover_red, wavelet_type, dwt_level)
    cover_dwt_green = multi_level_dwt(cover_green, wavelet_type, dwt_level)
    cover_dwt_blue = multi_level_dwt(cover_blue, wavelet_type, dwt_level)

    # Extract hidden DWT coefficients for each RGB channel
    extracted_hidden_dwt_red = extract_from_highest_level_approximation(
        stego_dwt_red,
        hidden_u_matrices_red,
        hidden_vt_matrices_red,
        embed_scale,
        hidden_dwt_red[0].shape,
    )
    extracted_hidden_dwt_green = extract_from_highest_level_approximation(
        stego_dwt_green,
        hidden_u_matrices_green,
        hidden_vt_matrices_green,
        embed_scale,
        hidden_dwt_green[0].shape,
    )
    extracted_hidden_dwt_blue = extract_from_highest_level_approximation(
        stego_dwt_blue,
        hidden_u_matrices_blue,
        hidden_vt_matrices_blue,
        embed_scale,
        hidden_dwt_blue[0].shape,
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
