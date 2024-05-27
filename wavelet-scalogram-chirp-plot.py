import numpy as np
import matplotlib.pyplot as plt
import pywt

# Generate a chirp signal
t = np.linspace(0, 1, 1000)  # Time vector from 0 to 1 second
f0 = 0  # Starting frequency in Hz
f1 = 30  # Ending frequency in Hz
chirp_signal = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / 2 * t**2))

# # Define wavelet parameters
wavelet = 'cmor1.5-1.0'  # Complex Morlet wavelet
scales = np.arange(1, 128)  # Wavelet scales

# Compute wavelet scalogram
coefficients, frequencies = pywt.cwt(chirp_signal, scales, wavelet)

# Plot the chirp signal
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, chirp_signal)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Sinal Chirp')

# Plot the wavelet scalogram
plt.subplot(2, 1, 2)
plt.imshow(np.abs(coefficients), extent=[0, 1, 0, 30], aspect='auto', cmap='jet', 
           vmax=np.max(np.abs(coefficients)), vmin=0)
plt.colorbar(label='Magnitude')
plt.xlabel('Tempo (s)')
plt.ylabel('Frequência (Hz)')
plt.title('Escalograma Wavelet do Sinal Chirp')


plt.tight_layout()
plt.show()