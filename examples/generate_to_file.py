#!/usr/bin/env python3

from deepfakeecg import generate
import os

# Create output directory if it doesn't exist
output_dir = "generated_ecgs"
os.makedirs(output_dir, exist_ok=True)

# Generate 5 ECG samples starting from ID 0
generate(
    num_of_sample=5,
    out_dir=output_dir,
    start_id=0
)

print(f"Generated 5 ECG samples in {output_dir}")
print("Files generated:")
for file in os.listdir(output_dir)[:5]:
    print(f" - {file}")
