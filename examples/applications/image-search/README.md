# Image Search
SentenceTransformers provides models that allow to embed images and text into the same vector space. This allows to find similar images as well as to implement **image search**.


![ImageSearch](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/ImageSearch.png)


## Installation
Ensure that you have [torchvision](https://pypi.org/project/torchvision/) installed to use the image-text-models and use a recent PyTorch version (tested with PyTorch 1.7.0). Image-Text-Models have been added with SentenceTransformers version 1.0.0. Image-Text-Models are still in an experimental phase. 

## Usage
SentenceTransformers provides a wrapper for the [OpenAI CLIP Model](https://github.com/openai/CLIP), which was trained on a variety of (image, text)-pairs.

```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image

#Load CLIP model
model = SentenceTransformer('clip-ViT-B-32')

#Encode an image:
img_emb = model.encode(Image.open('two_dogs_in_snow.jpg'))

#Encode text descriptions
text_emb = model.encode(['Two dogs in the snow', 'A cat on a table', 'A picture of London at night'])

#Compute cosine similarities 
cos_scores = util.cos_sim(img_emb, text_emb)
print(cos_scores)
```

You can use the CLIP model for:
- Text-to-Image / Image-To-Text / Image-to-Image / Text-to-Text Search
- You can fine-tune it on your own image&text data with the regular SentenceTransformers training code. 

## Examples
- [Image_Search.ipynb](Image_Search.ipynb) depicts a larger example for image-search using 25,000 free pictures from [Unsplash](https://unsplash.com/).