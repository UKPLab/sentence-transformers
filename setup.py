from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()



setup(
    name="sentence-transformers",
    version="2.2.2",
    author="Nils Reimers",
    author_email="info@nils-reimers.de",
    description="Multilingual text embeddings",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://www.SBERT.net",
    download_url="https://github.com/UKPLab/sentence-transformers/",
    packages=find_packages(),
    python_requires=">=3.6.0",
    install_requires=[
        'transformers>=4.6.0,<5.0.0',
        'tqdm',
        'torch>=1.6.0',
        'torchvision',
        'numpy',
        'scikit-learn',
        'scipy',
        'nltk',
        'sentencepiece',
        'huggingface-hub>=0.4.0'
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords="Transformer Networks BERT XLNet sentence embedding PyTorch NLP deep learning"
)
