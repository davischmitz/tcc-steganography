import subprocess

# Define the paths to the scripts
steganography_script = "steganography_higher_band.py"
metrics_script = "metrics.py"

# Define the wavelet types to test
wavelet_types = ["haar", "db1", "db8", "db16", "sym2", "sym8", "sym16"]

cover_image_path = "lena_std.tif"
hidden_image_path = "qrcode_compact_512x512.tif"
stego_image_path = "stego_image.tif"

# Define the embedding scale
embed_scale = 0.02  # You can adjust this value as needed

for wavelet_type in wavelet_types:
    print(f"Running steganography script with wavelet type: {wavelet_type}...")
    steganography_process = subprocess.run(
        [
            "python3",
            steganography_script,
            "--cover_image",
            cover_image_path,
            "--hidden_image",
            hidden_image_path,
            "--wavelet_type",
            wavelet_type,
            "--embed_scale",
            str(embed_scale),
        ],
        check=True,
    )
    print(f"Steganography script completed for wavelet type: {wavelet_type}.")

    print("Running metrics script...")
    metrics_process = subprocess.run(
        [
            "python3",
            metrics_script,
            "--wavelet_type",
            wavelet_type,
            "--original_image",
            cover_image_path,
            "--stego_image",
            "stego.tif",
        ],
        check=True,
    )
    print("Metrics script completed.")
