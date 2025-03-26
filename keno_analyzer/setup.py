from setuptools import find_packages, setup

setup(
    name="keno_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "plotly>=5.3.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for analyzing and optimizing Keno strategies",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="keno, analysis, strategy, optimization",
    url="https://github.com/yourusername/keno_analyzer",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "keno=keno.cli:main",
        ],
    },
)
