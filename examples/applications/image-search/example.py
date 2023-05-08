from sentence_transformers import SentenceTransformer, util, models
from PIL import ImageFile, Image
import numpy as np
import requests




###########

image = Image.open('two_dogs_in_snow.jpg')

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")



inputs = processor(texts=["a cat", "a dog"], images=[image], return_tensors="pt", padding=True)
output = model(**inputs)
#vision_outputs = model.vision_model(pixel_values=inputs['pixel_values'])
#image_embeds = model.visual_projection(vision_outputs[1])

#print(image_embeds.shape)
#exit()



#Load CLIP model
clip = models.CLIPModel()
model = SentenceTransformer(modules=[clip])

model.save('tmp-clip-model')

model = SentenceTransformer('tmp-clip-model')

#Encode an image:
img_emb = model.encode(Image.open('two_dogs_in_snow.jpg'))

#Encode text descriptions
text_emb = model.encode(['Two dogs in the snow', 'A cat on a table', 'A picture of London at night'])

#Compute cosine similarities
cos_scores = util.cos_sim(img_emb, text_emb)
print(cos_scores)