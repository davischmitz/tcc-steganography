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


def embed_svd(cover_svd, hidden_svd, scale):
    """Embed hidden image's singular values into the cover image's SVD using a scale factor."""
    cover_u, cover_s, cover_vh = cover_svd
    hidden_u, hidden_s, hidden_vh = hidden_svd
    embedded_s = cover_s + scale * hidden_s
    return embedded_s


def apply_svd(matrix):
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    return u, singular_values, vh


def decode_svd(embedded_matrix, u, vh):
    return np.dot(u * embedded_matrix, vh)


def extract_hidden_svd(embedded_svd, scale):
    return embedded_svd / scale


def display_image_subplot(image, position, title, rows, cols):
    plt.subplot(rows, cols, position)
    plt.axis("off")
    plt.title(title)
    plt.imshow(image, aspect="equal")


def display_dwt_coefficients(coefficients, position_start, rows, cols, labels):
    for i, coefficient in enumerate(coefficients):
        plt.subplot(rows, cols, position_start + i)
        plt.title(labels[i])
        plt.axis("off")
        plt.imshow(coefficient, interpolation="nearest", cmap=plt.cm.gray)


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


# def display_multi_level_dwt_coefficients(coefficients):
#     pos = 1  # Initialize position for subplots

#     # First element is the approximation (LL) at the highest level
#     ll = coefficients[0]
#     plt.subplot(len(coefficients), 4, pos)
#     plt.title(f"Level {len(coefficients) - 1} Approximation (Low Freq)")
#     plt.axis("off")
#     plt.imshow(ll, interpolation="nearest", cmap=plt.cm.gray)
#     pos += 4  # Move to the next row for detail coefficients

#     # Remaining elements contain LH, HL, HH detail coefficients
#     for level in range(1, len(coefficients)):
#         lh, hl, hh = coefficients[level]  # Unpack the detail coefficients

#         plt.subplot(len(coefficients), 4, pos)
#         plt.title(f"Level {len(coefficients) - level} LH (High Freq)")
#         plt.axis("off")
#         plt.imshow(lh, interpolation="nearest", cmap=plt.cm.gray)
#         pos += 1

#         plt.subplot(len(coefficients), 4, pos)
#         plt.title(f"Level {len(coefficients) - level} HL (High Freq)")
#         plt.axis("off")
#         plt.imshow(hl, interpolation="nearest", cmap=plt.cm.gray)
#         pos += 1

#         plt.subplot(len(coefficients), 4, pos)
#         plt.title(f"Level {len(coefficients) - level} HH (High Freq)")
#         plt.axis("off")
#         plt.imshow(hh, interpolation="nearest", cmap=plt.cm.gray)
#         pos += 1


def save_image(image, file_path, size=None):
    fig = plt.figure(frameon=False)
    if size is not None:
        fig.set_size_inches(size[0], size[1])
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    # ax.imshow(image, aspect="auto")
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
    rows, cols = 6, 4
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

    # Display cover and hidden images
    display_image_subplot(cover_image, 2, "Cover Image", rows, cols)
    display_image_subplot(hidden_image, 3, "Image to Hide", rows, cols)

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

    # # Apply DWT on RGB channels of both cover and hidden images
    # cover_dwt_red = compute_dwt(cover_red, wavelet_type)
    # cover_dwt_green = compute_dwt(cover_green, wavelet_type)
    # cover_dwt_blue = compute_dwt(cover_blue, wavelet_type)

    # hidden_dwt_red = compute_dwt(hidden_red, wavelet_type)
    # hidden_dwt_green = compute_dwt(hidden_green, wavelet_type)
    # hidden_dwt_blue = compute_dwt(hidden_blue, wavelet_type)

    # Display DWT coefficients
    # dwt_labels = ["Approximation", "Horizontal", "Vertical", "Diagonal"]
    # display_dwt_coefficients(cover_dwt_red, 5, rows, cols, dwt_labels)
    # display_dwt_coefficients(hidden_dwt_red, 9, rows, cols, dwt_labels)

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

    # # Apply SVD on the cover image's high-frequency bands (Horizontal, Vertical, Diagonal)
    # cover_svd_red = [apply_svd(band) for band in cover_dwt_red[1:]]
    # cover_svd_green = [apply_svd(band) for band in cover_dwt_green[1:]]
    # cover_svd_blue = [apply_svd(band) for band in cover_dwt_blue[1:]]

    # # Apply SVD on the hidden image's high-frequency bands
    # hidden_svd_red = [apply_svd(band) for band in hidden_dwt_red[1:]]
    # hidden_svd_green = [apply_svd(band) for band in hidden_dwt_green[1:]]
    # hidden_svd_blue = [apply_svd(band) for band in hidden_dwt_blue[1:]]

    # # Embed hidden image into cover image's SVD D values
    # embedded_red = [
    #     cover_svd_red[i][1] + embed_scale * hidden_svd_red[i][1] for i in range(3)
    # ]
    # embedded_green = [
    #     cover_svd_green[i][1] + embed_scale * hidden_svd_green[i][1] for i in range(3)
    # ]
    # embedded_blue = [
    #     cover_svd_blue[i][1] + embed_scale * hidden_svd_blue[i][1] for i in range(3)
    # ]

    # # Reconstruct DWT coefficients using embedded SVD data
    # embedded_dwt_red = (
    #     cover_dwt_red[0],
    #     tuple(
    #         np.dot(cover_svd_red[i][0] * embedded_red[i], cover_svd_red[i][2])
    #         for i in range(3)
    #     ),
    # )
    # embedded_dwt_green = (
    #     cover_dwt_green[0],
    #     tuple(
    #         np.dot(cover_svd_green[i][0] * embedded_green[i], cover_svd_green[i][2])
    #         for i in range(3)
    #     ),
    # )
    # embedded_dwt_blue = (
    #     cover_dwt_blue[0],
    #     tuple(
    #         np.dot(cover_svd_blue[i][0] * embedded_blue[i], cover_svd_blue[i][2])
    #         for i in range(3)
    #     ),
    # )

    # # Inverse DWT to get stego image
    # stego_red = compute_idwt(embedded_dwt_red[0], embedded_dwt_red[1], wavelet_type)
    # stego_green = compute_idwt(
    #     embedded_dwt_green[0], embedded_dwt_green[1], wavelet_type
    # )
    # stego_blue = compute_idwt(embedded_dwt_blue[0], embedded_dwt_blue[1], wavelet_type)

    # Merge stego channels to form final stego image
    stego_image = cv2.merge(
        (stego_red.astype(int), stego_green.astype(int), stego_blue.astype(int))
    )

    # Display stego image
    display_image_subplot(stego_image, 14, "Stego Image", rows, cols)

    ####################################################
    # DECODING PROCESS
    ####################################################

    # # Apply DWT to the stego image to get the high-frequency bands
    # stego_red_dwt = compute_dwt(stego_image[:, :, 0], wavelet_type)
    # stego_green_dwt = compute_dwt(stego_image[:, :, 1], wavelet_type)
    # stego_blue_dwt = compute_dwt(stego_image[:, :, 2], wavelet_type)

    # # Apply SVD on the stego image's high-frequency bands
    # stego_svd_red = [apply_svd(band) for band in stego_red_dwt[1:]]
    # stego_svd_green = [apply_svd(band) for band in stego_green_dwt[1:]]
    # stego_svd_blue = [apply_svd(band) for band in stego_blue_dwt[1:]]

    # # Extract hidden singular values by subtracting the cover image's singular values
    # extracted_hidden_svd_red = [
    #     extract_hidden_svd(stego_svd_red[i][1], embed_scale) for i in range(3)
    # ]
    # extracted_hidden_svd_green = [
    #     extract_hidden_svd(stego_svd_green[i][1], embed_scale) for i in range(3)
    # ]
    # extracted_hidden_svd_blue = [
    #     extract_hidden_svd(stego_svd_blue[i][1], embed_scale) for i in range(3)
    # ]

    # # Reconstruct the hidden image's DWT coefficients
    # reconstructed_hidden_dwt_red = (
    #     hidden_dwt_red[0],
    #     tuple(
    #         decode_svd(
    #             extracted_hidden_svd_red[i], hidden_svd_red[i][0], hidden_svd_red[i][2]
    #         )
    #         for i in range(3)
    #     ),
    # )
    # reconstructed_hidden_dwt_green = (
    #     hidden_dwt_green[0],
    #     tuple(
    #         decode_svd(
    #             extracted_hidden_svd_green[i],
    #             hidden_svd_green[i][0],
    #             hidden_svd_green[i][2],
    #         )
    #         for i in range(3)
    #     ),
    # )
    # reconstructed_hidden_dwt_blue = (
    #     hidden_dwt_blue[0],
    #     tuple(
    #         decode_svd(
    #             extracted_hidden_svd_blue[i],
    #             hidden_svd_blue[i][0],
    #             hidden_svd_blue[i][2],
    #         )
    #         for i in range(3)
    #     ),
    # )

    # # Perform Inverse DWT to get the extracted hidden image
    # extracted_hidden_red = compute_idwt(
    #     reconstructed_hidden_dwt_red[0], reconstructed_hidden_dwt_red[1], wavelet_type
    # )
    # extracted_hidden_green = compute_idwt(
    #     reconstructed_hidden_dwt_green[0],
    #     reconstructed_hidden_dwt_green[1],
    #     wavelet_type,
    # )
    # extracted_hidden_blue = compute_idwt(
    #     reconstructed_hidden_dwt_blue[0], reconstructed_hidden_dwt_blue[1], wavelet_type
    # )

    # # Merge RGB channels to form the extracted hidden image
    # extracted_hidden_image = cv2.merge(
    #     (
    #         extracted_hidden_red.astype(int),
    #         extracted_hidden_green.astype(int),
    #         extracted_hidden_blue.astype(int),
    #     )
    # )

    # # Display extracted hidden image
    # display_image_subplot(
    #     extracted_hidden_image, 15, "Extracted Hidden Image", rows, cols
    # )
    plt.show()

    # Save stego image
    save_image(
        stego_image,
        stego_image_path,
        size=(float(stego_image.shape[1]) / 100, float(stego_image.shape[0]) / 100),
    )

    # # Save extracted hidden image
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