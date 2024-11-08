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


def compress_hidden_image(hidden_image, rank):
    """Apply SVD to the hidden image and compress by selecting only the first 'rank' components."""
    hidden_u, hidden_s, hidden_vt = apply_svd(hidden_image)
    compressed_u = hidden_u[:, :rank]
    compressed_s = hidden_s[:rank]
    compressed_vt = hidden_vt[:rank, :]
    compressed_hidden = np.dot(
        compressed_u, np.dot(np.diag(compressed_s), compressed_vt)
    )
    return compressed_hidden


def resize_image_to_match(image, target_shape):
    """Resize the image to match the target shape for embedding."""
    resized_image = resize(image, target_shape, anti_aliasing=True)
    return resized_image


def apply_svd(matrix):
    """Perform SVD on the given matrix."""
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    return u, s, vt


def embed_svd(cover, hidden, scale):
    return cover + scale * hidden


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


# def embed_compressed_in_hh_band(cover_dwt, compressed_hidden, scale):
#     """Embed the compressed hidden image into the HH sub-band of the highest DWT level."""
#     # Get the high-frequency HH sub-band of the highest level
#     _, _, cover_hh = cover_dwt[-1]  # HH is the third element in the tuple at the highest level

#     # Resize the compressed hidden image to match the cover HH sub-band size
#     compressed_hidden_resized = resize_image_to_match(compressed_hidden, cover_hh.shape)

#     plt.imshow(compressed_hidden_resized, cmap="gray")
#     plt.show()

#     # Apply SVD on the HH sub-band of the cover and the compressed hidden image
#     cover_u, cover_s, cover_vt = apply_svd(cover_hh)
#     hidden_u, hidden_s, hidden_vt = apply_svd(compressed_hidden_resized)

#     # Embed compressed hidden image's singular values into the cover's singular values
#     embedded_u = embed_svd(cover_u, hidden_u, scale)
#     embedded_s = embed_svd(cover_s, hidden_s, scale)
#     embedded_vt = embed_svd(cover_vt, hidden_vt, scale)

#     # Reconstruct the embedded HH sub-band
#     embedded_hh = np.dot(embedded_u, np.dot(np.diag(embedded_s), embedded_vt))

#     # Replace the HH sub-band with the embedded HH sub-band in the highest level
#     cover_dwt[-1] = (cover_dwt[-1][0], cover_dwt[-1][1], embedded_hh)
#     return cover_dwt


def embed_all_bands_in_highest_level(cover_dwt, hidden_dwt, scale):
    """Embed all bands from the highest DWT level of the hidden image into the corresponding bands of the cover image."""
    # Identify the highest level bands in both cover and hidden images
    cover_ll, (cover_hl, cover_lh, cover_hh) = cover_dwt[0], cover_dwt[1]
    hidden_ll, (hidden_hl, hidden_lh, hidden_hh) = hidden_dwt[0], hidden_dwt[1]

    # Embed using SVD for each corresponding band (LL, HL, LH, HH)
    embedded_ll = embed_svd_in_band_limited(cover_ll, hidden_ll, scale)
    embedded_hl = embed_svd_in_band_limited(cover_hl, hidden_hl, scale)
    embedded_lh = embed_svd_in_band_limited(cover_lh, hidden_lh, scale)
    embedded_hh = embed_svd_in_band_limited(cover_hh, hidden_hh, scale)

    # # Reconstruct the modified DWT coefficients with embedded bands at the highest level
    # embedded_dwt = [embedded_ll, (embedded_hl, embedded_lh, embedded_hh)]
    # # Append the rest of the DWT coefficients from the cover image
    # embedded_dwt.extend(cover_dwt[2:])
    embedded_dwt = [embedded_ll]
    embedded_dwt.extend(cover_dwt[1:])
    return embedded_dwt


def embed_svd_in_band(cover_band, hidden_band, scale):
    """Embed hidden image in a single band using SVD."""
    cover_u, cover_s, cover_vt = apply_svd(cover_band)
    hidden_u, hidden_s, hidden_vt = apply_svd(hidden_band)

    # Embed hidden image's singular values into the cover's singular values
    embedded_u = embed_svd(cover_u, hidden_u, scale)
    embedded_s = embed_svd(cover_s, hidden_s, scale)
    embedded_vt = embed_svd(cover_vt, hidden_vt, scale)

    # Reconstruct the embedded band
    embedded_band = np.dot(embedded_u, np.dot(np.diag(embedded_s), embedded_vt))
    return embedded_band


def embed_svd_in_band_limited(cover_band, hidden_band, scale, preserve_rank=100):
    """Embed the first 'preserve_rank' components of the hidden image into the cover, starting from the 101st rank of the cover."""
    # Perform SVD on both the cover and hidden bands
    cover_u, cover_s, cover_vt = apply_svd(cover_band)
    hidden_u, hidden_s, hidden_vt = apply_svd(hidden_band)

    # Initialize embedded matrices by copying the cover's SVD components
    embedded_u = np.copy(cover_u)
    embedded_s = np.copy(cover_s)
    embedded_vt = np.copy(cover_vt)

    # Embed the first 'preserve_rank' components of the hidden image into the cover, starting at the 101st rank
    embedded_u[:, preserve_rank : preserve_rank * 2] += (
        scale * hidden_u[:, :preserve_rank]
    )
    embedded_s[preserve_rank : preserve_rank * 2] += scale * hidden_s[:preserve_rank]
    embedded_vt[preserve_rank : preserve_rank * 2, :] += (
        scale * hidden_vt[:preserve_rank, :]
    )

    # Reconstruct the embedded band using the modified matrices
    embedded_band = np.dot(embedded_u, np.dot(np.diag(embedded_s), embedded_vt))
    return embedded_band


def decode_all_bands_from_highest_level(stego_dwt, cover_dwt, scale):
    """Decode the hidden image from all bands in the highest DWT level of the stego image, using the cover image's DWT coefficients."""
    # Extract the highest level bands from the stego and cover images
    stego_ll = stego_dwt[0]  # LL band at the highest level
    stego_hl, stego_lh, stego_hh = stego_dwt[
        1
    ]  # Detail coefficients (HL, LH, HH) at the highest level

    cover_ll = cover_dwt[0]  # LL band of the cover image at the highest level
    cover_hl, cover_lh, cover_hh = cover_dwt[
        1
    ]  # Detail coefficients (HL, LH, HH) of the cover image at the highest level

    # Decode each band by reversing the embedding
    decoded_ll = decode_svd_in_band_limited(stego_ll, cover_ll, scale)
    decoded_hl = decode_svd_in_band_limited(stego_hl, cover_hl, scale)
    decoded_lh = decode_svd_in_band_limited(stego_lh, cover_lh, scale)
    decoded_hh = decode_svd_in_band_limited(stego_hh, cover_hh, scale)

    # Return the decoded DWT coefficients as a list structured like the original hidden DWT
    decoded_dwt = [decoded_ll]
    # decoded_dwt.extend(cover_dwt[2:])
    return decoded_dwt


def decode_svd_in_band(stego_band, cover_band, scale):
    """Decode the hidden band from a single stego band using SVD, with access to the cover band."""
    # Perform SVD on both the stego and cover bands
    stego_u, stego_s, stego_vt = apply_svd(stego_band)
    cover_u, cover_s, cover_vt = apply_svd(cover_band)

    # Retrieve the hidden image's U, S, and Vt matrices by isolating the embedded component
    hidden_u = (stego_u - cover_u) / scale
    hidden_s = (stego_s - cover_s) / scale
    hidden_vt = (stego_vt - cover_vt) / scale

    # Reconstruct the hidden band using the decoded singular values
    decoded_band = np.dot(hidden_u, np.dot(np.diag(hidden_s), hidden_vt))
    return decoded_band


def decode_svd_in_band_limited(stego_band, cover_band, scale, preserve_rank=100):
    """Decode the hidden band from a single stego band using SVD, with access to the cover band, starting from the 101st rank."""
    # Perform SVD on both the stego and cover bands
    stego_u, stego_s, stego_vt = apply_svd(stego_band)
    cover_u, cover_s, cover_vt = apply_svd(cover_band)

    # Extract the hidden image's U, S, and Vt matrices by isolating the embedded components from the 101st rank onward
    hidden_u = (
        stego_u[:, preserve_rank : preserve_rank * 2]
        - cover_u[:, preserve_rank : preserve_rank * 2]
    ) / scale
    hidden_s = (
        stego_s[preserve_rank : preserve_rank * 2]
        - cover_s[preserve_rank : preserve_rank * 2]
    ) / scale
    hidden_vt = (
        stego_vt[preserve_rank : preserve_rank * 2, :]
        - cover_vt[preserve_rank : preserve_rank * 2, :]
    ) / scale

    # Reconstruct the hidden band using the decoded matrices
    decoded_band = np.dot(hidden_u, np.dot(np.diag(hidden_s), hidden_vt))
    return decoded_band


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

    # Compress the hidden image
    compressed_hidden_red = compress_hidden_image(hidden_red, compression_ranks)
    compressed_hidden_green = compress_hidden_image(hidden_green, compression_ranks)
    compressed_hidden_blue = compress_hidden_image(hidden_blue, compression_ranks)

    # Apply multi-level DWT on RGB channels of both cover and hidden images
    cover_dwt_red = multi_level_dwt(cover_red, wavelet_type, dwt_level)
    cover_dwt_green = multi_level_dwt(cover_green, wavelet_type, dwt_level)
    cover_dwt_blue = multi_level_dwt(cover_blue, wavelet_type, dwt_level)

    hidden_dwt_red = multi_level_dwt(hidden_red, wavelet_type, dwt_level)
    hidden_dwt_green = multi_level_dwt(hidden_green, wavelet_type, dwt_level)
    hidden_dwt_blue = multi_level_dwt(hidden_blue, wavelet_type, dwt_level)

    # Display the coefficients
    display_multi_level_dwt_coefficients(cover_dwt_red, dwt_level, dwt_level)
    display_multi_level_dwt_coefficients(hidden_dwt_red, dwt_level, dwt_level)

    stego_dwt_red = embed_all_bands_in_highest_level(
        cover_dwt_red, hidden_dwt_red, embed_scale
    )
    stego_dwt_green = embed_all_bands_in_highest_level(
        cover_dwt_green, hidden_dwt_green, embed_scale
    )
    stego_dwt_blue = embed_all_bands_in_highest_level(
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
    stego_dwt_red = multi_level_dwt(stego_red, wavelet_type, dwt_level)
    stego_dwt_green = multi_level_dwt(stego_green, wavelet_type, dwt_level)
    stego_dwt_blue = multi_level_dwt(stego_blue, wavelet_type, dwt_level)

    cover_dwt_red = multi_level_dwt(cover_red, wavelet_type, dwt_level)
    cover_dwt_green = multi_level_dwt(cover_green, wavelet_type, dwt_level)
    cover_dwt_blue = multi_level_dwt(cover_blue, wavelet_type, dwt_level)

    # Extract hidden DWT coefficients for each RGB channel
    extracted_hidden_dwt_red = decode_all_bands_from_highest_level(
        stego_dwt_red, cover_dwt_red, embed_scale
    )
    extracted_hidden_dwt_green = decode_all_bands_from_highest_level(
        stego_dwt_green, cover_dwt_green, embed_scale
    )
    extracted_hidden_dwt_blue = decode_all_bands_from_highest_level(
        stego_dwt_blue, cover_dwt_blue, embed_scale
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
