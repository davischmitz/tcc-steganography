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

    # Apply DWT on RGB channels of both cover and hidden images
    cover_dwt_red = compute_dwt(cover_red, wavelet_type)
    cover_dwt_green = compute_dwt(cover_green, wavelet_type)
    cover_dwt_blue = compute_dwt(cover_blue, wavelet_type)

    hidden_dwt_red = compute_dwt(hidden_red, wavelet_type)
    hidden_dwt_green = compute_dwt(hidden_green, wavelet_type)
    hidden_dwt_blue = compute_dwt(hidden_blue, wavelet_type)

    # Display DWT coefficients
    dwt_labels = ["Approximation", "Horizontal", "Vertical", "Diagonal"]
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
    display_image_subplot(stego_image, 14, "Stego Image", rows, cols)

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
        extracted_hidden_image, 15, "Extracted Hidden Image", rows, cols
    )

    plt.show()

    # 11) Apply the decoding transform to each channel of the stego image
    # applying dwt to 3 stego channel images to get coeffs of stego image in R,G,B

    # Psend_r = pywt.dwt2(processed_rgbr, decoding_wavelet)
    # PcAr, (PcHr, PcVr, PcDr) = Psend_r

    # Psend_g = pywt.dwt2(processed_rgbg, decoding_wavelet)
    # PcAg, (PcHg, PcVg, PcDg) = Psend_g

    # Psend_b = pywt.dwt2(processed_rgbb, decoding_wavelet)
    # PcAb, (PcHb, PcVb, PcDb) = Psend_b

    # # 12) Perform Singular Value Decomposition (SVD) on the stego image
    # PPr, PDr, PQr = np.linalg.svd(PcAr, full_matrices=False)
    # PPg, PDg, PQg = np.linalg.svd(PcAg, full_matrices=False)
    # PPb, PDb, PQb = np.linalg.svd(PcAb, full_matrices=False)

    # # 13) Reverse the information embedded in the 'D' parameter of the cover image in step 5 through the inverse operation
    # # subtract from R,G,B channels values of cover image
    # S_ewatr = (PDr - Dr) / 0.1
    # S_ewatg = (PDg - Dg) / 0.1
    # S_ewatb = (PDb - Db) / 0.1

    # # 14) Combine the approximations with the hidden SVD values to reconstruct the hidden image
    # ewatr = np.dot(P1r * S_ewatr, Q1r)
    # ewatg = np.dot(P1g * S_ewatg, Q1g)
    # ewatb = np.dot(P1b * S_ewatb, Q1b)

    # # 15) Obtain the reconstructed hidden image, which consists of the color channels combined with the normalized SVD differences
    # d = ewatr.astype(int)
    # e = ewatg.astype(int)
    # f = ewatb.astype(int)
    # eimg = cv2.merge((d, e, f))

    # # 16) Extract the horizontal, vertical, and diagonal coefficients from each RGB channel of the hidden image
    # # coeffs of original hidden image except the new derived appproximation
    # eproc_r = eimg[:, :, 0], (cHr1, cVr1, cDr1)
    # eproc_g = eimg[:, :, 1], (cHg1, cVg1, cDg1)
    # eproc_b = eimg[:, :, 2], (cHb1, cVb1, cDb1)

    # # 17) Apply inverse transform to each channel of the image to generate the final hidden information image
    # # hidden stego images get high res r,g,b seperate images/channels usign idwt
    # eprocessed_rgbr = pywt.idwt2(eproc_r, decoding_wavelet)
    # eprocessed_rgbg = pywt.idwt2(eproc_g, decoding_wavelet)
    # eprocessed_rgbb = pywt.idwt2(eproc_b, decoding_wavelet)

    # # just converting float to int prior to cv2.merge
    # x1 = eprocessed_rgbr.astype(int)

    # y1 = eprocessed_rgbg.astype(int)

    # z1 = eprocessed_rgbb.astype(int)

    # # 18) combine different high res r,g,b to get hidden image
    # hidden_rgb = cv2.merge((x1, y1, z1))

    # h1, w1, ch1 = hidden_rgb.shape

    # plt.subplot(plot_rows, plot_columns, 15)
    # plt.axis("off")
    # plt.title("Decoded Hidden Image")
    # plt.imshow(hidden_rgb, aspect="equal")

    # plt.show()
    # plt.close()

    # # save stego image to filesystem
    # save_image_to_file(wimghd, "stego.tif", figsize=(float(w) / 100, float(h) / 100))

    # # save decoded hidden image to filesystem
    # save_image_to_file(
    #     hidden_rgb, "hidden_rgb.tif", figsize=(float(w) / 100, float(h) / 100)
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

    # Compare cover and stego images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cover_image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(stego_image, cmap="gray")
    axes[1].set_title("Stego Image")
    axes[1].axis("off")

    # plt.show()


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
