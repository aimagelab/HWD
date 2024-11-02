from setuptools import setup, find_packages

setup(
    name="hwd",
    version="1.0",
    author="Vittorio Pippi",
    author_email="vittorio.pippi@unimore.it",
    description="Module which contains all scores nedded for text comparison",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aimagelab/HWD",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[    # Add dependencies here
        'torch',
        'torchvision',
        'torchmetrics',
        'scikit-learn',
        'opencv-python',
        'tqdm',
        'transformers',
        'tiktoken',
        'protobuf',
    ],
)
