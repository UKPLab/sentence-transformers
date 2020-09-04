from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()



setup(
    name="sentence-transformers",
    version="0.3.6",
    author="Nils Reimers",
    author_email="info@nils-reimers.de",
    description="Sentence Embeddings using BERT / RoBERTa / XLNet",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://github.com/UKPLab/sentence-transformers",
    download_url="https://github.com/UKPLab/sentence-transformers/archive/v0.3.5.zip",
    packages=find_packages(),
    install_requires=[
        'transformers>=3.1.0,<3.2.0',
        'tqdm',
        'torch>=1.2.0',
        'numpy',
        'scikit-learn',
        'scipy',
        'nltk'
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords="Transformer Networks BERT XLNet sentence embedding PyTorch NLP deep learning"
)
