import numpy as np
import pywt
from matplotlib import pyplot as plt
import cv2
from skimage.transform import resize

####################################################
# ENCODING PROCESS
####################################################


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


def save_image(image, file_path, size=None):
    fig = plt.figure(frameon=False)
    if size is not None:
        fig.set_size_inches(size[0], size[1])
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, aspect="auto")
    fig.savefig(file_path)


# Configuration
wavelet_type = "haar"
embed_scale = 0.02
rows, cols = 4, 4

# Load images
cover_image = load_and_convert_to_rgb("lena_std.tif")
hidden_image = load_and_convert_to_rgb("qrcode_compact_512x512.tif")


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

# Apply SVD on the cover image's high-frequency bands (Horizontal, Vertical, Diagonal)
cover_svd_red = [apply_svd(band) for band in cover_dwt_red[1:]]
cover_svd_green = [apply_svd(band) for band in cover_dwt_green[1:]]
cover_svd_blue = [apply_svd(band) for band in cover_dwt_blue[1:]]

# Apply SVD on the hidden image's high-frequency bands
hidden_svd_red = [apply_svd(band) for band in hidden_dwt_red[1:]]
hidden_svd_green = [apply_svd(band) for band in hidden_dwt_green[1:]]
hidden_svd_blue = [apply_svd(band) for band in hidden_dwt_blue[1:]]

# Embed hidden image into cover image's SVD D values
embedded_red = [
    cover_svd_red[i][1] + embed_scale * hidden_svd_red[i][1] for i in range(3)
]
embedded_green = [
    cover_svd_green[i][1] + embed_scale * hidden_svd_green[i][1] for i in range(3)
]
embedded_blue = [
    cover_svd_blue[i][1] + embed_scale * hidden_svd_blue[i][1] for i in range(3)
]

# Reconstruct DWT coefficients using embedded SVD data
embedded_dwt_red = (
    cover_dwt_red[0],
    tuple(
        np.dot(cover_svd_red[i][0] * embedded_red[i], cover_svd_red[i][2])
        for i in range(3)
    ),
)
embedded_dwt_green = (
    cover_dwt_green[0],
    tuple(
        np.dot(cover_svd_green[i][0] * embedded_green[i], cover_svd_green[i][2])
        for i in range(3)
    ),
)
embedded_dwt_blue = (
    cover_dwt_blue[0],
    tuple(
        np.dot(cover_svd_blue[i][0] * embedded_blue[i], cover_svd_blue[i][2])
        for i in range(3)
    ),
)

# Inverse DWT to get stego image
stego_red = compute_idwt(embedded_dwt_red[0], embedded_dwt_red[1], wavelet_type)
stego_green = compute_idwt(embedded_dwt_green[0], embedded_dwt_green[1], wavelet_type)
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


def decode_svd(embedded_matrix, u, vh):
    return np.dot(u * embedded_matrix, vh)


def extract_hidden_svd(embedded_svd, cover_svd, scale):
    return (embedded_svd - cover_svd) / scale


# Apply DWT to the stego image to get the high-frequency bands
stego_red_dwt = compute_dwt(stego_image[:, :, 0], wavelet_type)
stego_green_dwt = compute_dwt(stego_image[:, :, 1], wavelet_type)
stego_blue_dwt = compute_dwt(stego_image[:, :, 2], wavelet_type)

# Apply SVD on the stego image's high-frequency bands
stego_svd_red = [apply_svd(band) for band in stego_red_dwt[1:]]
stego_svd_green = [apply_svd(band) for band in stego_green_dwt[1:]]
stego_svd_blue = [apply_svd(band) for band in stego_blue_dwt[1:]]

# Extract hidden singular values by subtracting the cover image's singular values
extracted_hidden_svd_red = [
    extract_hidden_svd(stego_svd_red[i][1], cover_svd_red[i][1], embed_scale)
    for i in range(3)
]
extracted_hidden_svd_green = [
    extract_hidden_svd(stego_svd_green[i][1], cover_svd_green[i][1], embed_scale)
    for i in range(3)
]
extracted_hidden_svd_blue = [
    extract_hidden_svd(stego_svd_blue[i][1], cover_svd_blue[i][1], embed_scale)
    for i in range(3)
]

# Reconstruct the hidden image's DWT coefficients
reconstructed_hidden_dwt_red = (
    hidden_dwt_red[0],
    tuple(
        decode_svd(
            extracted_hidden_svd_red[i], hidden_svd_red[i][0], hidden_svd_red[i][2]
        )
        for i in range(3)
    ),
)
reconstructed_hidden_dwt_green = (
    hidden_dwt_green[0],
    tuple(
        decode_svd(
            extracted_hidden_svd_green[i],
            hidden_svd_green[i][0],
            hidden_svd_green[i][2],
        )
        for i in range(3)
    ),
)
reconstructed_hidden_dwt_blue = (
    hidden_dwt_blue[0],
    tuple(
        decode_svd(
            extracted_hidden_svd_blue[i], hidden_svd_blue[i][0], hidden_svd_blue[i][2]
        )
        for i in range(3)
    ),
)

# Perform Inverse DWT to get the extracted hidden image
extracted_hidden_red = compute_idwt(
    reconstructed_hidden_dwt_red[0], reconstructed_hidden_dwt_red[1], wavelet_type
)
extracted_hidden_green = compute_idwt(
    reconstructed_hidden_dwt_green[0], reconstructed_hidden_dwt_green[1], wavelet_type
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
display_image_subplot(extracted_hidden_image, 15, "Extracted Hidden Image", rows, cols)
plt.show()

# Save stego image
save_image(
    stego_image,
    "stego.tif",
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
