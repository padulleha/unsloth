# setup.py for unsloth - Fast LLM finetuning library
# Fork of unslothai/unsloth with additional improvements

from setuptools import setup, find_packages
import os

# Read the README for the long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Core dependencies required for the package
INSTALL_REQUIRES = [
    "torch>=2.1.0",
    "transformers>=4.38.0",
    "datasets>=2.16.0",
    "sentencepiece>=0.1.99",
    "tqdm>=4.66.0",
    "psutil",
    "wheel>=0.42.0",
    "packaging>=23.1",
    "tyro>=0.5.0",
]

# Optional dependencies for extended functionality
EXTRAS_REQUIRE = {
    "training": [
        "trl>=0.7.10",
        "peft>=0.9.0",
        "accelerate>=0.27.0",
        "bitsandbytes>=0.43.0",
        "xformers>=0.0.24",
    ],
    "vision": [
        "Pillow>=9.0.0",
        "torchvision>=0.16.0",
    ],
    "export": [
        "gguf>=0.6.0",
        "llama-cpp-python>=0.2.0",
    ],
    # Added numpy here since I kept running into missing numpy errors during dev
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "numpy>=1.24.0",
        # Added ipykernel so I can use this in Jupyter notebooks without hassle
        "ipykernel>=6.0.0",
        # matplotlib is handy for plotting training loss curves during experiments
        "matplotlib>=3.7.0",
        # wandb makes it easy to track runs across different experiments
        "wandb>=0.16.0",
    ],
}

# Convenience alias to install everything
EXTRAS_REQUIRE["all"] = [
    dep
    for group, deps in EXTRAS_REQUIRE.items()
    if group != "dev"
    for dep in deps
]

setup(
    name="unsloth",
    version="2024.12.0",
    author="Unsloth AI",
    author_email="info@unsloth.ai",
    description="2-5x faster, 70% less memory LLM finetuning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/unslothai/unsloth",
    project_urls={
        "Bug Tracker": "https://github.com/unslothai/unsloth/issues",
        "Documentation": "https://docs.unsloth.ai",
        "Source Code": "https://github.com/unslothai/unsloth",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "l
