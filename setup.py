from setuptools import setup

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="sentence-transformers",
    version="0.1.0",
    author="Nils Reimers, Gregor Geigle",
    author_email="reimers@ukp.informatik.tu-darmstadt.de",
    desciption="PyTorch implementation of Sentence Transformers (BERT, XLNet) for sentence embeddings",
    long_description=readme,
    license="Apache License 2.0",
    url="https://github.com/UKPLab/sentence-transformers",
    packages=["sentence_transformers", "sentence_transformers.models"],
    install_requires=[
        "pytorch-transformers==1.0.0",
        "tqdm",
        "torch>=1.0.1",
        "numpy",
        "scikit-learn",
        "scipy"
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
