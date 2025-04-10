#!/usr/bin/env python3

import matplotlib.pyplot as plt
import deepfakeecg
import numpy as np

# Generate a single ECG sample
ecg_data = deepfakeecg.generate_as_numpy()

# Plot the first lead (Lead I)
plt.figure(figsize=(15, 3))
plt.plot(ecg_data[:, 0])
plt.title("Generated ECG - Lead I")
plt.xlabel("Sample")
plt.ylabel("Amplitude (μV)")
plt.grid(True)

# Print shape and basic stats
print(f"ECG data shape: {ecg_data.shape}")
print(f"Value range: [{np.min(ecg_data)}, {np.max(ecg_data)}]")
print("\nLead order: [I, II, V1, V2, V3, V4, V5, V6]")

plt.show()
