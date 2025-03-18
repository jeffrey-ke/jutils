from setuptools import setup, find_packages

setup(
    name="jutils",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "tensorboard"
    ],
    author="Jeffrey Ke",
    author_email="jke3@andrew.cmu.edu",
    description="jutils",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",  # Change if uploading to GitHub
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)