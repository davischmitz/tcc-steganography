import numpy as np
import pywt
from matplotlib import pyplot as plt
import cv2
from skimage.transform import resize
import argparse
import os
import csv
import math


def load_and_convert_to_rgb(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def display_image_subplot(image, position, title, rows, cols):
    plt.subplot(rows, cols, position)
    plt.axis("off")
    plt.title(title)
    plt.imshow(image, aspect="equal")


def compute_dwt(image, wavelet_type):
    approximation, (horizontal, vertical, diagonal) = pywt.dwt2(image, wavelet_type)
    return approximation, horizontal, vertical, diagonal


def compute_idwt(approximation, details, wavelet_type):
    return pywt.idwt2((approximation, details), wavelet_type)


def display_dwt_coefficients(coefficients, position_start, rows, cols, labels):
    for i, coefficient in enumerate(coefficients):
        plt.subplot(rows, cols, position_start + i)
        plt.title(labels[i])
        plt.axis("off")
        plt.imshow(coefficient, interpolation="nearest", cmap=plt.cm.gray)


def apply_svd(matrix):
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    return u, singular_values, vh


def decode_svd(embedded_matrix, u, vh):
    return np.dot(u * embedded_matrix, vh)


def extract_hidden_svd(embedded_svd, cover_svd, scale):
    return (embedded_svd - cover_svd) / scale


def save_image(image, file_path, size=None):
    fig = plt.figure(frameon=False)
    if size is not None:
        fig.set_size_inches(size[0], size[1])
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, aspect="auto")
    fig.savefig(file_path)


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
    rows, cols = 4, 4

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
    display_image_subplot(cover_image, 2, "Imagem Cover", rows, cols)
    display_image_subplot(hidden_image, 3, "Imagem Embedded", rows, cols)

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

    # Apply DWT on RGB channels of both cover and hidden images
    cover_dwt_red = compute_dwt(cover_red, wavelet_type)
    cover_dwt_green = compute_dwt(cover_green, wavelet_type)
    cover_dwt_blue = compute_dwt(cover_blue, wavelet_type)

    hidden_dwt_red = compute_dwt(hidden_red, wavelet_type)
    hidden_dwt_green = compute_dwt(hidden_green, wavelet_type)
    hidden_dwt_blue = compute_dwt(hidden_blue, wavelet_type)

    # Display DWT coefficients
    dwt_labels = [
        "Aproximação",
        "Detalhes Horizontais",
        "Detalhes Verticais",
        "Detalhes Diagonais",
    ]
    display_dwt_coefficients(cover_dwt_red, 5, rows, cols, dwt_labels)
    display_dwt_coefficients(hidden_dwt_red, 9, rows, cols, dwt_labels)

    # Apply SVD on the cover image's low frequency band (Approximation)
    cover_svd_red = apply_svd(cover_dwt_red[0])
    cover_svd_green = apply_svd(cover_dwt_green[0])
    cover_svd_blue = apply_svd(cover_dwt_blue[0])

    # Apply SVD on the hidden image's low frequency band (Approximation)
    hidden_svd_red = apply_svd(hidden_dwt_red[0])
    hidden_svd_green = apply_svd(hidden_dwt_green[0])
    hidden_svd_blue = apply_svd(hidden_dwt_blue[0])

    # Embed hidden image into cover image's SVD D values
    embedded_red = [cover_svd_red[1] + embed_scale * hidden_svd_red[1]]
    embedded_green = [cover_svd_green[1] + embed_scale * hidden_svd_green[1]]
    embedded_blue = [cover_svd_blue[1] + embed_scale * hidden_svd_blue[1]]

    # Reconstruct DWT coefficients using embedded SVD data
    embedded_dwt_red = (
        np.dot(cover_svd_red[0] * embedded_red, cover_svd_red[2]),
        cover_dwt_red[1:],
    )
    embedded_dwt_green = (
        np.dot(cover_svd_green[0] * embedded_green, cover_svd_green[2]),
        cover_dwt_green[1:],
    )
    embedded_dwt_blue = (
        np.dot(cover_svd_blue[0] * embedded_blue, cover_svd_blue[2]),
        cover_dwt_blue[1:],
    )

    # Inverse DWT to get stego image
    stego_red = compute_idwt(embedded_dwt_red[0], embedded_dwt_red[1], wavelet_type)
    stego_green = compute_idwt(
        embedded_dwt_green[0], embedded_dwt_green[1], wavelet_type
    )
    stego_blue = compute_idwt(embedded_dwt_blue[0], embedded_dwt_blue[1], wavelet_type)

    # Merge stego channels to form final stego image
    stego_image = cv2.merge(
        (stego_red.astype(int), stego_green.astype(int), stego_blue.astype(int))
    )

    # Display stego image
    display_image_subplot(stego_image, 14, "Imagem Stego", rows, cols)

    ####################################################
    # DECODING PROCESS
    ####################################################

    # Apply DWT to the stego image to get the apporximation and details
    stego_red_dwt = compute_dwt(stego_image[:, :, 0], wavelet_type)
    stego_green_dwt = compute_dwt(stego_image[:, :, 1], wavelet_type)
    stego_blue_dwt = compute_dwt(stego_image[:, :, 2], wavelet_type)

    # Apply SVD on the stego image's low-frequency band (Approximation)
    stego_svd_red = apply_svd(stego_red_dwt[0])
    stego_svd_green = apply_svd(stego_green_dwt[0])
    stego_svd_blue = apply_svd(stego_blue_dwt[0])

    # Extract hidden singular values by subtracting the cover image's singular values
    extracted_hidden_svd_red = [
        extract_hidden_svd(stego_svd_red[1], cover_svd_red[1], embed_scale)
    ]
    extracted_hidden_svd_green = [
        extract_hidden_svd(stego_svd_green[1], cover_svd_green[1], embed_scale)
    ]
    extracted_hidden_svd_blue = [
        extract_hidden_svd(stego_svd_blue[1], cover_svd_blue[1], embed_scale)
    ]

    # Reconstruct the hidden image's DWT coefficients
    reconstructed_hidden_dwt_red = (
        decode_svd(extracted_hidden_svd_red, hidden_svd_red[0], hidden_svd_red[2]),
        stego_red_dwt[1:],
    )
    reconstructed_hidden_dwt_green = (
        decode_svd(
            extracted_hidden_svd_green, hidden_svd_green[0], hidden_svd_green[2]
        ),
        stego_green_dwt[1:],
    )
    reconstructed_hidden_dwt_blue = (
        decode_svd(extracted_hidden_svd_blue, hidden_svd_blue[0], hidden_svd_blue[2]),
        stego_blue_dwt[1:],
    )

    # Perform Inverse DWT to get the extracted hidden image
    extracted_hidden_red = compute_idwt(
        reconstructed_hidden_dwt_red[0], reconstructed_hidden_dwt_red[1], wavelet_type
    )
    extracted_hidden_green = compute_idwt(
        reconstructed_hidden_dwt_green[0],
        reconstructed_hidden_dwt_green[1],
        wavelet_type,
    )
    extracted_hidden_blue = compute_idwt(
        reconstructed_hidden_dwt_blue[0], reconstructed_hidden_dwt_blue[1], wavelet_type
    )

    # Merge RGB channels to form the extracted hidden image
    extracted_hidden_image = cv2.merge(
        (
            extracted_hidden_red.astype(int),
            extracted_hidden_green.astype(int),
            extracted_hidden_blue.astype(int),
        )
    )

    # Display extracted hidden image
    display_image_subplot(
        extracted_hidden_image, 15, "Imagem Embedded extraída", rows, cols
    )

    plot_histograms_with_matplotlib(cover_image, stego_image)

    plt.show()

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
                ]
            )
        writer.writerow(
            [
                "Lower Band",
                cover_image_path,
                hidden_image_path,
                cover_image.shape,
                stego_image.shape,
                wavelet_type,
                embed_scale,
                mse_value,
                psnr_value,
                psnr_value_hidden_extracted,
            ]
        )

    print(f"Results saved to {csv_filename}")

    # Compare cover and stego images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cover_image, cmap="gray")
    axes[0].set_title("Imagem Cover")
    axes[0].axis("off")

    axes[1].imshow(stego_image, cmap="gray")
    axes[1].set_title("Imagem Stego")
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
    args = parser.parse_args()

    main(
        args.wavelet_type,
        args.embed_scale,
        args.cover_image,
        args.hidden_image,
        args.stego_image,
    )
