import spectral_entropy
import numpy as np

# Example time series data
time_series = np.random.randn(1000)
Fs = 1000  # Sampling frequency in Hz

# Calculate spectral entropy
spectral_entropy_values, time_vector = spectral_entropy.calculate_spectral_entropy(time_series, Fs)

# Print the results
print("Spectral Entropy Values:", spectral_entropy_values)
print("Time Vector:", time_vector)


