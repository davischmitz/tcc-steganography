import numpy as np
import pywt
from matplotlib import pyplot as plt
import cv2
from skimage.transform import resize


def read_and_convert_image(image_path):
    # 1) reading files
    image = cv2.imread(image_path)
    # 2) converting to rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def show_image_subplot(image, subplot_number, title):
    plt.subplot(plot_rows, plot_columns, subplot_number)
    plt.axis("off")
    plt.title(title)
    plt.imshow(image, aspect="equal")


def dwt_coefficients(image, encoding_wavelet):
    cA, (cH, cV, cD) = pywt.dwt2(image, encoding_wavelet)
    return cA, cH, cV, cD


def show_coefficients_subplot(coefficients, subplot_base, dwt_labels):
    for i, a in enumerate(coefficients):
        subplot_number = subplot_base + i
        plt.subplot(plot_rows, plot_columns, subplot_number)
        plt.title(dwt_labels[i])
        plt.axis("off")
        plt.imshow(a, interpolation="nearest", cmap=plt.cm.gray)


def svd_decomposition(matrix):
    P, D, Q = np.linalg.svd(matrix, full_matrices=False)
    return P, D, Q


def save_image_to_file(image, filepath, figsize=None):
    fig = plt.figure(frameon=False)
    if figsize is not None:
        fig.set_size_inches(figsize[0], figsize[1])
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, aspect="auto")
    fig.savefig(filepath)


encoding_wavelet = "haar"
decoding_wavelet = "haar"

plot_rows = 4
plot_columns = 4

to_hide_og = read_and_convert_image("qrcode_compact_512x512.tif")
to_send_og = read_and_convert_image("lena_std.tif")

# Crop the images to be the same dimensions
min_height = min(to_hide_og.shape[0], to_send_og.shape[0])
min_width = min(to_hide_og.shape[1], to_send_og.shape[1])

to_hide_og = to_hide_og[:min_height, :min_width]
to_send_og = to_send_og[:min_height, :min_width]

# plot cover and image that will be hidden
show_image_subplot(to_send_og, 2, "Cover Image")
show_image_subplot(to_hide_og, 3, "Image to hide")

# --------------------------------------------
# Encoding Process
# Process of hiding an image within another image
# ---------------------------------------------

dimh, dimw, dimch = to_send_og.shape

# 3) seperating channels (colors) for cover and hidden images
to_send_r = to_send_og[:, :, 0]
to_send_g = to_send_og[:, :, 1]
to_send_b = to_send_og[:, :, 2]

to_hide_r = to_hide_og[:, :, 0]
to_hide_g = to_hide_og[:, :, 1]
to_hide_b = to_hide_og[:, :, 2]

# 4) Apply the wavelet transform to the cover and hidden images
cAr, cHr, cVr, cDr = dwt_coefficients(to_send_r, encoding_wavelet)
cAg, cHg, cVg, cDg = dwt_coefficients(to_send_g, encoding_wavelet)
cAb, cHb, cVb, cDb = dwt_coefficients(to_send_b, encoding_wavelet)

cAr1, cHr1, cVr1, cDr1 = dwt_coefficients(to_hide_r, encoding_wavelet)
cAg1, cHg1, cVg1, cDg1 = dwt_coefficients(to_hide_g, encoding_wavelet)
cAb1, cHb1, cVb1, cDb1 = dwt_coefficients(to_hide_b, encoding_wavelet)

# plot all layers resulted from DWT
dwt_labels = [
    "Approximation",
    "Horizontal Detail",
    "Vertical Detail",
    "Diagonal Detail",
]

show_coefficients_subplot([cAr, cHr, cVr, cDr], 5, dwt_labels)
show_coefficients_subplot([cAr1, cHr1, cVr1, cDr1], 9, dwt_labels)

print(cAr.shape)

# # 5) Perform Singular Value Decomposition (SVD) on the cover and hidden images

# Pr, Dr, Qr = svd_decomposition(cAr)
# Pg, Dg, Qg = svd_decomposition(cAg)
# Pb, Db, Qb = svd_decomposition(cAb)

# P1r, D1r, Q1r = svd_decomposition(cAr1)
# P1g, D1g, Q1g = svd_decomposition(cAg1)
# P1b, D1b, Q1b = svd_decomposition(cAb1)

# print(Pr.shape, Dr.shape, Qr.shape)  # just for debugging

# # 6) Embed the hidden information into the 'D' parameters of the cover image

# S_wimgr = Dr + (0.02 * D1r)
# S_wimgg = Dg + (0.02 * D1g)
# S_wimgb = Db + (0.02 * D1b)

# # 7) Reconstruct the coefficient matrix from the embedded SVD parameters

# wimgr = np.dot(Pr * S_wimgr, Qr)

# wimgg = np.dot(Pg * S_wimgg, Qg)

# wimgb = np.dot(Pb * S_wimgb, Qb)

# 5) Perform Singular Value Decomposition (SVD) on the cover image's high-frequency bands (Horizontal, Vertical, Diagonal)

# Red channel
Pr_h, Dr_h, Qr_h = svd_decomposition(cHr)
Pr_v, Dr_v, Qr_v = svd_decomposition(cVr)
Pr_d, Dr_d, Qr_d = svd_decomposition(cDr)

# Green channel
Pg_h, Dg_h, Qg_h = svd_decomposition(cHg)
Pg_v, Dg_v, Qg_v = svd_decomposition(cVg)
Pg_d, Dg_d, Qg_d = svd_decomposition(cDg)

# Blue channel
Pb_h, Db_h, Qb_h = svd_decomposition(cHb)
Pb_v, Db_v, Qb_v = svd_decomposition(cVb)
Pb_d, Db_d, Qb_d = svd_decomposition(cDb)

# Perform SVD on the hidden image's high-frequency bands
P1r_h, D1r_h, Q1r_h = svd_decomposition(cHr1)
P1r_v, D1r_v, Q1r_v = svd_decomposition(cVr1)
P1r_d, D1r_d, Q1r_d = svd_decomposition(cDr1)

P1g_h, D1g_h, Q1g_h = svd_decomposition(cHg1)
P1g_v, D1g_v, Q1g_v = svd_decomposition(cVg1)
P1g_d, D1g_d, Q1g_d = svd_decomposition(cDg1)

P1b_h, D1b_h, Q1b_h = svd_decomposition(cHb1)
P1b_v, D1b_v, Q1b_v = svd_decomposition(cVb1)
P1b_d, D1b_d, Q1b_d = svd_decomposition(cDb1)

# 6) Embed the hidden information into the 'D' parameters of the high-frequency bands

# Red channel embedding in high-frequency bands
S_wimgr_h = Dr_h + (0.02 * D1r_h)
S_wimgr_v = Dr_v + (0.02 * D1r_v)
S_wimgr_d = Dr_d + (0.02 * D1r_d)

# Green channel embedding in high-frequency bands
S_wimgg_h = Dg_h + (0.02 * D1g_h)
S_wimgg_v = Dg_v + (0.02 * D1g_v)
S_wimgg_d = Dg_d + (0.02 * D1g_d)

# Blue channel embedding in high-frequency bands
S_wimgb_h = Db_h + (0.02 * D1b_h)
S_wimgb_v = Db_v + (0.02 * D1b_v)
S_wimgb_d = Db_d + (0.02 * D1b_d)

# 7) Reconstruct the coefficient matrices for high-frequency bands from the embedded SVD parameters

# Reconstruct for Red channel
wimgr_h = np.dot(Pr_h * S_wimgr_h, Qr_h)
wimgr_v = np.dot(Pr_v * S_wimgr_v, Qr_v)
wimgr_d = np.dot(Pr_d * S_wimgr_d, Qr_d)

# Reconstruct for Green channel
wimgg_h = np.dot(Pg_h * S_wimgg_h, Qg_h)
wimgg_v = np.dot(Pg_v * S_wimgg_v, Qg_v)
wimgg_d = np.dot(Pg_d * S_wimgg_d, Qg_d)

# Reconstruct for Blue channel
wimgb_h = np.dot(Pb_h * S_wimgb_h, Qb_h)
wimgb_v = np.dot(Pb_v * S_wimgb_v, Qb_v)
wimgb_d = np.dot(Pb_d * S_wimgb_d, Qb_d)

# At this point, you've modified the high-frequency coefficients with the embedded data
# You can now use these modified high-frequency bands to apply the inverse DWT and reconstruct the stego image


# cast type from merged r, g, b
# a = wimgr.astype(int)  # red matrix
# b = wimgg.astype(int)  # green matrix
# c = wimgb.astype(int)  # blue matrix


# 8) Concatenate the three reconstructed RGB channels into a single matrix
# wimg = cv2.merge((a, b, c))
# h, w, ch = wimg.shape

# 9) Extract the horizontal, vertical, and diagonal coefficients from each RGB channel of the image
# rgb coeffs for idwt, so that you can recreate a original img but with cA now having hidden info
# (cHr, cVr, cDr) -> wavelet coefficients corresponding to the horizontal (cHr), vertical (cVr) and diagonal (cDr) components obtained from the wavelet transform of the red channel
proc_r = cAr, (wimgr_h, wimgr_v, wimgr_d)
proc_g = cAg, (wimgg_h, wimgg_v, wimgg_d)
proc_b = cAb, (wimgb_h, wimgb_v, wimgb_d)

# 10) Apply inverse transform to each channel of the processed image, generating the stego image
processed_rgbr = pywt.idwt2(proc_r, encoding_wavelet)
processed_rgbg = pywt.idwt2(proc_g, encoding_wavelet)
processed_rgbb = pywt.idwt2(proc_b, encoding_wavelet)
# reconstructed color channels were obtained, in which the modified wavelet coefficients were incorporated back
# they represent the steganographed image, in which the hidden information is embedded in the modified wavelet coefficients

# combine color channels into a single image
wimghd = cv2.merge(
    (processed_rgbr.astype(int), processed_rgbg.astype(int), processed_rgbb.astype(int))
)

h, w, ch = wimghd.shape

# plot stego image
plt.subplot(plot_rows, plot_columns, 14)
plt.axis("off")
plt.title("Stego Image")
plt.imshow(wimghd, aspect="equal")

# --------------------------------------------
# Decoding Process
# Steganography reversal process using high-frequency details
# ---------------------------------------------

# 11) Apply the decoding transform to each channel of the stego image
# applying dwt to 3 stego channel images to get high-frequency coeffs of stego image in R, G, B

Psend_r = pywt.dwt2(processed_rgbr, decoding_wavelet)
_, (PcHr, PcVr, PcDr) = (
    Psend_r  # Ignore approximation, use only high-frequency components
)

Psend_g = pywt.dwt2(processed_rgbg, decoding_wavelet)
_, (PcHg, PcVg, PcDg) = Psend_g

Psend_b = pywt.dwt2(processed_rgbb, decoding_wavelet)
_, (PcHb, PcVb, PcDb) = Psend_b

# 12) Perform Singular Value Decomposition (SVD) on the high-frequency bands of the stego image

# Red channel high-frequency details
PPr_h, PDr_h, PQr_h = np.linalg.svd(PcHr, full_matrices=False)
PPr_v, PDr_v, PQr_v = np.linalg.svd(PcVr, full_matrices=False)
PPr_d, PDr_d, PQr_d = np.linalg.svd(PcDr, full_matrices=False)

# Green channel high-frequency details
PPg_h, PDg_h, PQg_h = np.linalg.svd(PcHg, full_matrices=False)
PPg_v, PDg_v, PQg_v = np.linalg.svd(PcVg, full_matrices=False)
PPg_d, PDg_d, PQg_d = np.linalg.svd(PcDg, full_matrices=False)

# Blue channel high-frequency details
PPb_h, PDb_h, PQb_h = np.linalg.svd(PcHb, full_matrices=False)
PPb_v, PDb_v, PQb_v = np.linalg.svd(PcVb, full_matrices=False)
PPb_d, PDb_d, PQb_d = np.linalg.svd(PcDb, full_matrices=False)

# 13) Reverse the embedded information in the 'D' parameter of the high-frequency bands (using the same scaling factor as in encoding)
# subtract from the SVD of stego image high-frequency bands, and divide by the scale factor

# Red channel high-frequency
S_ewatr_h = (PDr_h - Dr_h) / 0.02
S_ewatr_v = (PDr_v - Dr_v) / 0.02
S_ewatr_d = (PDr_d - Dr_d) / 0.02

# Green channel high-frequency
S_ewatg_h = (PDg_h - Dg_h) / 0.02
S_ewatg_v = (PDg_v - Dg_v) / 0.02
S_ewatg_d = (PDg_d - Dg_d) / 0.02

# Blue channel high-frequency
S_ewatb_h = (PDb_h - Db_h) / 0.02
S_ewatb_v = (PDb_v - Db_v) / 0.02
S_ewatb_d = (PDb_d - Db_d) / 0.02

# 14) Combine the extracted high-frequency SVD components with the hidden SVD matrices to reconstruct the hidden image's high-frequency details

# Red channel reconstruction
ewatr_h = np.dot(P1r_h * S_ewatr_h, Q1r_h)
ewatr_v = np.dot(P1r_v * S_ewatr_v, Q1r_v)
ewatr_d = np.dot(P1r_d * S_ewatr_d, Q1r_d)

# Green channel reconstruction
ewatg_h = np.dot(P1g_h * S_ewatg_h, Q1g_h)
ewatg_v = np.dot(P1g_v * S_ewatg_v, Q1g_v)
ewatg_d = np.dot(P1g_d * S_ewatg_d, Q1g_d)

# Blue channel reconstruction
ewatb_h = np.dot(P1b_h * S_ewatb_h, Q1b_h)
ewatb_v = np.dot(P1b_v * S_ewatb_v, Q1b_v)
ewatb_d = np.dot(P1b_d * S_ewatb_d, Q1b_d)

# 15) Reconstruct the hidden image by applying the inverse DWT to the extracted high-frequency details

# Reconstruct the wavelet coefficients for each channel of the hidden image (keep the original approximation cA from the hidden image)
eproc_r = cAr1, (ewatr_h, ewatr_v, ewatr_d)
eproc_g = cAg1, (ewatg_h, ewatg_v, ewatg_d)
eproc_b = cAb1, (ewatb_h, ewatb_v, ewatb_d)

# Apply inverse DWT to obtain the reconstructed hidden image channels
eprocessed_rgbr = pywt.idwt2(eproc_r, decoding_wavelet)
eprocessed_rgbg = pywt.idwt2(eproc_g, decoding_wavelet)
eprocessed_rgbb = pywt.idwt2(eproc_b, decoding_wavelet)

# Convert float to int
x1 = eprocessed_rgbr.astype(int)
y1 = eprocessed_rgbg.astype(int)
z1 = eprocessed_rgbb.astype(int)

# 16) Combine the reconstructed RGB channels into the final hidden image
hidden_rgb = cv2.merge((x1, y1, z1))

# Display the decoded hidden image
plt.subplot(plot_rows, plot_columns, 15)
plt.axis("off")
plt.title("Decoded Hidden Image")
plt.imshow(hidden_rgb, aspect="equal")

plt.show()
plt.close()

# save stego image to filesystem
save_image_to_file(wimghd, "stego.tif", figsize=(float(w) / 100, float(h) / 100))

# save decoded hidden image to filesystem
save_image_to_file(
    hidden_rgb, "hidden_rgb.tif", figsize=(float(w) / 100, float(h) / 100)
)
