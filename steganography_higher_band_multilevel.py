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


def compute_multi_level_dwt(image, wavelet_type, level):
    return pywt.wavedec2(image, wavelet_type, level=level)


def compute_multi_level_idwt(coeffs, wavelet_type):
    return pywt.waverec2(coeffs, wavelet_type)


def compute_inverse_multi_level_dwt(dwt_coeffs, wavelet_type):
    """
    Reconstructs an image from its multi-level DWT coefficients.

    Parameters:
    - dwt_coeffs: List of tuples containing DWT coefficients for each decomposition level.
                  The first element is the approximation coefficients at the coarsest level.
    - wavelet_type: The type of wavelet to use for the inverse transform (e.g., 'haar').

    Returns:
    - Reconstructed image from the provided DWT coefficients.
    """
    # Start with the approximation at the coarsest level
    current_coeffs = dwt_coeffs[0]

    # Iterate through each level in reverse (from coarsest to finest)
    for i in range(1, len(dwt_coeffs)):
        # current_coeffs is the approximation, dwt_coeffs[i] are the details for this level
        current_coeffs = pywt.idwt2((current_coeffs, dwt_coeffs[i]), wavelet_type)

    return current_coeffs


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


# def modify_high_freq_bands_for_multilevel(cover_coeffs, hidden_coeffs, embed_scale):
#     """
#     Modify the high-frequency details of each DWT level using SVD.
#     """
#     approx_cover, details_cover = cover_coeffs[0], cover_coeffs[1:]
#     approx_hidden, details_hidden = hidden_coeffs[0], hidden_coeffs[1:]

#     modified_details = []
#     for cover_detail, hidden_detail in zip(details_cover, details_hidden):
#         modified_bands = []

#         # Embed each high-frequency sub-band (HH, HL, LH) separately
#         for cover_band, hidden_band in zip(cover_detail, hidden_detail):
#             cover_u, cover_s, cover_vt = np.linalg.svd(cover_band, full_matrices=False)
#             hidden_u, hidden_s, hidden_vt = np.linalg.svd(
#                 hidden_band, full_matrices=False
#             )

#             # Embed hidden singular values by scaling and adding to cover's singular values
#             embedded_s = cover_s + embed_scale * hidden_s

#             # Reconstruct the modified band with embedded singular values
#             modified_band = cover_u @ np.diag(embedded_s) @ cover_vt
#             modified_bands.append(modified_band)

#         modified_details.append(tuple(modified_bands))

#     return [approx_cover] + modified_details


# Embed only in the highest level's high-frequency bands (details of the last level)
def embed_in_highest_level_only_details(cover_dwt, hidden_dwt, scale):
    cover_details = cover_dwt[-1]  # High-frequency details at the highest level
    hidden_details = hidden_dwt[-1]  # High-frequency details of hidden image

    embedded_details = []
    for cover_band, hidden_band in zip(cover_details, hidden_details):
        cover_u, cover_s, cover_vt = apply_svd(cover_band)
        hidden_u, hidden_s, hidden_vt = apply_svd(hidden_band)
        embedded_s = embed_svd(cover_s, hidden_s, scale)
        embedded_band = np.dot(cover_u, np.dot(np.diag(embedded_s), cover_vt))
        embedded_details.append(embedded_band)

    # Replace only the highest level's high-frequency details with the embedded details
    modified_dwt = cover_dwt[:-1] + [tuple(embedded_details)]
    return modified_dwt


# Extract hidden image's details from the highest level's high-frequency bands
def extract_from_highest_level(stego_dwt, scale):
    stego_details = stego_dwt[-1]  # High-frequency details at the highest level

    extracted_details = []
    for stego_band in stego_details:
        stego_u, stego_s, stego_vt = apply_svd(stego_band)
        hidden_s = stego_s / scale  # Extract hidden singular values
        extracted_band = np.dot(stego_u, np.dot(np.diag(hidden_s), stego_vt))
        extracted_details.append(extracted_band)

    # Construct the hidden DWT coefficients with extracted details
    hidden_dwt = [stego_dwt[0]]  # Keep the approximation coefficients the same
    hidden_dwt.append(
        tuple(extracted_details)
    )  # Use extracted high-frequency bands for highest level

    # Initialize hidden DWT with zero approximation to remove dependency on stego approximation
    hidden_dwt = [np.zeros_like(stego_dwt[0])]  # Set approximation level to zeros

    # Extract high-frequency details from the highest level where data is embedded
    stego_details = stego_dwt[-1]

    extracted_details = []
    for stego_band in stego_details:
        stego_u, stego_s, stego_vt = apply_svd(stego_band)
        hidden_s = stego_s / scale  # Reverse the embedding by scaling down
        extracted_band = np.dot(stego_u, np.dot(np.diag(hidden_s), stego_vt))
        extracted_details.append(extracted_band)

    # Build the multi-level DWT structure with zeros for intermediate levels
    for _ in range(len(stego_dwt) - 2):
        hidden_dwt.append((None, None, None))

    # Add the extracted high-frequency details at the highest level as a 3-tuple
    hidden_dwt.append(tuple(extracted_details))
    return hidden_dwt


def extract_from_highest_level_only_details(stego_dwt, hidden_dwt, scale):
    stego_details = stego_dwt[
        -1
    ]  # High-frequency details at the highest level of stego image
    hidden_details = hidden_dwt[
        -1
    ]  # High-frequency details at the highest level of hidden image

    extracted_details = []
    for stego_band, hidden_band in zip(stego_details, hidden_details):
        # Apply SVD on both the stego and cover high-frequency bands
        stego_u, stego_s, stego_vt = apply_svd(stego_band)
        hidden_u, hidden_s, hidden_vt = apply_svd(hidden_band)

        # Extract hidden singular values from the difference in singular values
        extracted_hidden_svd = extract_hidden_svd(hidden_s, scale)

        # Reconstruct hidden image's high-frequency band
        reconstructed_hidden_band = np.dot(
            hidden_u, np.dot(np.diag(extracted_hidden_svd), hidden_vt)
        )
        extracted_details.append(reconstructed_hidden_band)

    # Replace the details in the highest level with the extracted hidden details
    return [hidden_dwt[0]] + [tuple(extracted_details)]


def decode_hidden_image_multi_detail(coeffs, levels):
    """
    Decodes the hidden image from a stego image encoded in all detail coefficients (LH, HL, HH)
    using multi-level DWT and SVD steganography.

    Parameters:
    stego_image (ndarray): The stego image containing the hidden image.
    wavelet (str): The wavelet type used for DWT (e.g., 'haar', 'db1').
    levels (int): The level of DWT decomposition used during encoding.

    Returns:
    hidden_image (ndarray): The extracted hidden image.
    """

    # Step 2: Initialize an empty list to hold the reconstructed hidden sub-bands
    hidden_details = []

    # Step 3: For each decomposition level, extract the hidden information from each detail band
    for i in range(1, levels + 1):
        LH, HL, HH = coeffs[i]

        # Decode the singular values from LH sub-band
        U_LH, S_LH_stego, Vt_LH = apply_svd(LH)
        hidden_LH = np.dot(U_LH, np.dot(np.diag(S_LH_stego), Vt_LH))

        # Decode the singular values from HL sub-band
        U_HL, S_HL_stego, Vt_HL = apply_svd(HL)
        hidden_HL = np.dot(U_HL, np.dot(np.diag(S_HL_stego), Vt_HL))

        # Decode the singular values from HH sub-band
        U_HH, S_HH_stego, Vt_HH = apply_svd(HH)
        hidden_HH = np.dot(U_HH, np.dot(np.diag(S_HH_stego), Vt_HH))

        # Store the reconstructed hidden sub-bands
        hidden_details.append((hidden_LH, hidden_HL, hidden_HH))

    # Step 4: Combine all hidden details and reconstruct the hidden image
    return [np.zeros_like(coeffs[0])] + hidden_details


# def embed_all_bands(cover_dwt, hidden_dwt, scale):
#     stego_dwt = cover_dwt.copy()

#     # Embed approximation coefficients of the hidden image into the details of the cover
#     cover_approx, hidden_approx = cover_dwt[0], hidden_dwt[0]
#     cover_u, cover_s, cover_vt = apply_svd(cover_approx)
#     hidden_u, hidden_s, hidden_vt = apply_svd(hidden_approx)
#     stego_s = embed_svd(cover_s, hidden_s, scale)
#     stego_dwt[0] = np.dot(cover_u, np.dot(np.diag(stego_s), cover_vt))

#     # Embed each detail coefficient of the hidden image into the corresponding detail coefficient of the cover
#     for i in range(1, len(cover_dwt)):
#         cover_details = cover_dwt[i]
#         hidden_details = hidden_dwt[i]

#         embedded_details = []
#         for cover_band, hidden_band in zip(cover_details, hidden_details):
#             # Perform SVD on both cover and hidden image bands
#             cover_u, cover_s, cover_vt = apply_svd(cover_band)
#             hidden_u, hidden_s, hidden_vt = apply_svd(hidden_band)

#             # Embed hidden image's singular values into cover image's singular values
#             stego_s = embed_svd(cover_s, hidden_s, scale)

#             # Reconstruct the embedded band
#             embedded_band = np.dot(cover_u, np.dot(np.diag(stego_s), cover_vt))
#             embedded_details.append(embedded_band)

#         # Add the embedded details as a tuple
#         stego_dwt[i] = tuple(embedded_details)

#     return stego_dwt


# def extract_all_bands(stego_dwt, cover_dwt, scale):
#     """
#     Extracts all the hidden image's coefficients (approximation and details) from the detail
#     coefficients of the stego image by reversing the SVD embedding.

#     Args:
#         stego_dwt (list): Multi-level DWT coefficients of the stego image.
#         cover_dwt (list): Multi-level DWT coefficients of the cover image.
#         scale (float): Scaling factor used during embedding.

#     Returns:
#         list: Multi-level DWT coefficients for the hidden image.
#     """
#     hidden_dwt = []

#     # Extract approximation coefficients of the hidden image from the approximation of stego and cover
#     stego_approx, cover_approx = stego_dwt[0], cover_dwt[0]
#     cover_u, cover_s, cover_vt = apply_svd(cover_approx)
#     stego_u, stego_s, stego_vt = apply_svd(stego_approx)
#     hidden_s = extract_svd(stego_s, cover_s, scale)
#     hidden_approx = np.dot(cover_u, np.dot(np.diag(hidden_s), cover_vt))
#     hidden_dwt.append(hidden_approx)

#     # Extract each detail coefficient of the hidden image from the corresponding detail coefficients of stego and cover
#     for i in range(1, len(stego_dwt)):
#         stego_details = stego_dwt[i]
#         cover_details = cover_dwt[i]

#         extracted_details = []
#         for stego_band, cover_band in zip(stego_details, cover_details):
#             # Perform SVD on both stego and cover bands
#             cover_u, cover_s, cover_vt = apply_svd(cover_band)
#             stego_u, stego_s, stego_vt = apply_svd(stego_band)

#             # Extract the hidden image's singular values
#             hidden_s = extract_svd(stego_s, cover_s, scale)

#             # Reconstruct the hidden high-frequency band
#             hidden_band = np.dot(cover_u, np.dot(np.diag(hidden_s), cover_vt))
#             extracted_details.append(hidden_band)

#         # Add the extracted details as a tuple
#         hidden_dwt.append(tuple(extracted_details))

#     return hidden_dwt


# def extract_hidden_from_dwt(stego_dwt, cover_dwt, embed_scale):
#     reconstructed_dwt = []
#     for level in range(
#         1, len(stego_dwt)
#     ):  # Start from level 1 to only get high-frequency bands
#         cover_details = cover_dwt[level]
#         stego_details = stego_dwt[level]
#         reconstructed_details = []

#         # For each band (HH, HL, LH), apply SVD extraction
#         for cover_band, stego_band in zip(cover_details, stego_details):
#             cover_u, cover_s, cover_vt = np.linalg.svd(cover_band, full_matrices=False)
#             stego_u, stego_s, stego_vt = np.linalg.svd(stego_band, full_matrices=False)

#             # Extract hidden singular values
#             hidden_s = (stego_s - cover_s) / embed_scale

#             # Reconstruct the hidden band from extracted singular values
#             hidden_band = cover_u @ np.diag(hidden_s) @ cover_vt
#             reconstructed_details.append(hidden_band)

#         reconstructed_dwt.append(tuple(reconstructed_details))

#     # Zero out the approximation coefficients for each level
#     reconstructed_dwt = [np.zeros_like(stego_dwt[0])] + reconstructed_dwt
#     return reconstructed_dwt


def extract_svd(stego_s, cover_s, scale):
    return (stego_s - cover_s) / scale


# def extract_hidden_high_freq_bands(stego_coeffs, cover_coeffs, embed_scale):
#     approx_stego, stego_details = stego_coeffs[0], stego_coeffs[1:]
#     approx_cover, cover_details = cover_coeffs[0], cover_coeffs[1:]

#     hidden_details = []
#     for stego_detail, cover_detail in zip(stego_details, cover_details):
#         hidden_bands = []
#         for stego_band, cover_band in zip(stego_detail, cover_detail):
#             stego_svd = apply_svd(stego_band)
#             cover_svd = apply_svd(cover_band)
#             extracted_s = extract_svd(stego_svd, cover_svd, embed_scale)
#             hidden_band = decode_svd(extracted_s, cover_svd[0], cover_svd[2])
#             hidden_bands.append(hidden_band)
#         hidden_details.append(tuple(hidden_bands))

#     return [approx_stego] + hidden_details


# def embed_svd(cover_svd, hidden_svd, scale):
#     cover_u, cover_s, cover_vh = cover_svd
#     hidden_u, hidden_s, hidden_vh = hidden_svd
#     embedded_s = cover_s + scale * hidden_s
#     return embedded_s


def embed_svd(cover_s, hidden_s, scale):
    return cover_s + scale * hidden_s


def apply_svd(matrix):
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    return u, singular_values, vh


def decode_svd(embedded_matrix, u, vh):
    return np.dot(u * embedded_matrix, vh)


# def extract_hidden_svd(stego_s, cover_s, scale):
#     hidden_s = (stego_s - cover_s) / scale
#     return hidden_s


def extract_hidden_svd(embedded_svd, scale):
    return embedded_svd / scale


# # Extract hidden image SVD from each level's high-frequency bands
# def extract_hidden_from_dwt(stego_dwt, embed_scale):
#     reconstructed_dwt = []

#     # Start from level 1 to only process high-frequency bands
#     for level in range(1, len(stego_dwt)):
#         stego_details = stego_dwt[level]
#         reconstructed_details = []

#         # For each band (HH, HL, LH), apply SVD extraction
#         for stego_band in stego_details:
#             stego_u, stego_s, stego_vt = np.linalg.svd(stego_band, full_matrices=False)

#             # Extract hidden singular values
#             hidden_s = extract_hidden_svd(stego_s, embed_scale)

#             # Reconstruct the hidden band from extracted singular values
#             hidden_band = stego_u @ np.diag(hidden_s) @ stego_vt
#             reconstructed_details.append(hidden_band)

#         reconstructed_dwt.append(tuple(reconstructed_details))

#     # Set approximation coefficients to zero for each level
#     reconstructed_dwt = [np.zeros_like(stego_dwt[0])] + reconstructed_dwt
#     return reconstructed_dwt


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
    dwt_level = 2

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
    modified_dwt_red = embed_in_highest_level_only_details(
        cover_dwt_red, hidden_dwt_red, embed_scale
    )
    modified_dwt_green = embed_in_highest_level_only_details(
        cover_dwt_green, hidden_dwt_green, embed_scale
    )
    modified_dwt_blue = embed_in_highest_level_only_details(
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
    stego_dwt_red = compute_multi_level_dwt(stego_red, wavelet_type, dwt_level)
    stego_dwt_green = compute_multi_level_dwt(stego_green, wavelet_type, dwt_level)
    stego_dwt_blue = compute_multi_level_dwt(stego_blue, wavelet_type, dwt_level)

    # cover_dwt_red = compute_multi_level_dwt(cover_red, wavelet_type, dwt_level)
    # cover_dwt_green = compute_multi_level_dwt(cover_green, wavelet_type, dwt_level)
    # cover_dwt_blue = compute_multi_level_dwt(cover_blue, wavelet_type, dwt_level)

    # Extract hidden DWT coefficients for each RGB channel
    extracted_hidden_dwt_red = decode_hidden_image_multi_detail(
        stego_dwt_red, dwt_level
    )
    extracted_hidden_dwt_green = decode_hidden_image_multi_detail(
        stego_dwt_green, dwt_level
    )
    extracted_hidden_dwt_blue = decode_hidden_image_multi_detail(
        stego_dwt_blue, dwt_level
    )

    # Perform Inverse DWT to get the hidden image
    extracted_hidden_red = compute_multi_level_idwt(
        extracted_hidden_dwt_red, wavelet_type
    )
    extracted_hidden_green = compute_multi_level_idwt(
        extracted_hidden_dwt_green, wavelet_type
    )
    extracted_hidden_blue = compute_multi_level_idwt(
        extracted_hidden_dwt_blue, wavelet_type
    )

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
