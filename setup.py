import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepfake-ecg", # Replace with your own username
    version="0.0.1",
    author="Vajira Thambawita",
    author_email="vlbthambawita@gmail.com",
    description="deepfake ECG generator for 10s long 8 or 12 leads ECGs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vlbthambawita/deepfake-ecg",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'tqdm',
        'torch',
        'torchvision',
        'pillow',
        'vector-quantize-pytorch>=0.1.0'
  ],
)