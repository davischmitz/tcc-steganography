import subprocess

# Define the paths to the scripts
steganography_script = "steganography_higher_band.py"
metrics_script = "metrics.py"

# Run the steganography script to generate the stego image
print("Running steganography script...")
steganography_process = subprocess.run(["python3", steganography_script], check=True)
print("Steganography script completed.")

# Run the metrics script to evaluate the generated images
print("Running metrics script...")
metrics_process = subprocess.run(["python3", metrics_script], check=True)
print("Metrics script completed.")
