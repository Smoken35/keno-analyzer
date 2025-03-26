"""Setup configuration for the Keno Analyzer package."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="keno-analyzer",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for analyzing Keno game patterns and predicting outcomes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Smoken35/keno-analyzer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "plotly>=5.3.0",
        "seaborn>=0.11.0",
        "matplotlib>=3.4.0",
        "beautifulsoup4>=4.9.0",
        "selenium>=4.0.0",
        "fake-useragent>=0.1.11",
        "webdriver-manager>=3.5.0",
        "backoff>=1.10.0",
        "schedule>=1.1.0",
        "tqdm>=4.62.0",
        "prometheus-client>=0.11.0",
        "PyYAML>=5.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.7b0",
            "isort>=5.9.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "types-PyYAML>=6.0.0",
        ],
    },
)
