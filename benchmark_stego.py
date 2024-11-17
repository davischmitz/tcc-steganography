import subprocess

low_frequency_stego_script = "steganography_lower_band.py"
high_frequency_stego_script = "steganography_higher_band.py"
multilevel_stego_script = "steganography_higher_band_multilevel_v2.py"

wavelet_types = ["haar", "db1", "db8", "db16", "sym2", "sym8", "sym16"]

cover_image_path = "peppers_color.tif"
hidden_image_path = "qrcode_512x512.tif"
stego_image_path = "stego.tif"

embedding_scales = [0.02, 0.05, 0.1, 0.2]

multilevel_dwt_embed_level = 2
multilevel_dwt_extract_level = 2

for wavelet_type in wavelet_types:
    for embedding_scale in embedding_scales:
        print(
            f"Running low frequency steganography script with wavelet type: {wavelet_type} and embedding scale: {embedding_scale}"
        )
        steganography_process = subprocess.run(
            [
                "python3",
                low_frequency_stego_script,
                "--cover_image",
                cover_image_path,
                "--hidden_image",
                hidden_image_path,
                "--stego_image",
                stego_image_path,
                "--wavelet_type",
                wavelet_type,
                "--embed_scale",
                str(embedding_scale),
            ],
            check=True,
        )
        print(
            f"Low frequency steganography script completed for wavelet type: {wavelet_type} and embedding scale: {embedding_scale}"
        )
        print(
            f"Running high frequency steganography script with wavelet type: {wavelet_type} and embedding scale: {embedding_scale}"
        )
        steganography_process = subprocess.run(
            [
                "python3",
                high_frequency_stego_script,
                "--cover_image",
                cover_image_path,
                "--hidden_image",
                hidden_image_path,
                "--stego_image",
                stego_image_path,
                "--wavelet_type",
                wavelet_type,
                "--embed_scale",
                str(embedding_scale),
            ],
            check=True,
        )
        print(
            f" High frequency steganography script completed for wavelet type: {wavelet_type} and embedding scale: {embedding_scale}"
        )
        print(
            f"Running multilevel steganography script with wavelet type: {wavelet_type} and embedding scale: {embedding_scale}"
        )
        steganography_process = subprocess.run(
            [
                "python3",
                multilevel_stego_script,
                "--cover_image",
                cover_image_path,
                "--hidden_image",
                hidden_image_path,
                "--stego_image",
                stego_image_path,
                "--wavelet_type",
                wavelet_type,
                "--embed_scale",
                str(embedding_scale),
                "--embed_dwt_level",
                str(multilevel_dwt_embed_level),
                "--extract_dwt_level",
                str(multilevel_dwt_extract_level),
            ],
            check=True,
        )
        print(
            f"Multilevel steganography script completed for wavelet type: {wavelet_type} and embedding scale: {embedding_scale}"
        )
