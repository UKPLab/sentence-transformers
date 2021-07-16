from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import torch
import pickle
import zipfile
from IPython.display import display
from IPython.display import Image as IPImage
import os
from tqdm.autonotebook import tqdm
import gradio as gr


# Here we load the multilingual CLIP model. Note, this model can only encode text.
# If you need embeddings for images, you must load the 'clip-ViT-B-32' model
model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')


# Next, we get about 25k images from Unsplash 
img_folder = 'photos/'
if not os.path.exists(img_folder) or len(os.listdir(img_folder)) == 0:
    os.makedirs(img_folder, exist_ok=True)
    
    photo_filename = 'unsplash-25k-photos.zip'
    if not os.path.exists(photo_filename):   #Download dataset if does not exist
        util.http_get('http://sbert.net/datasets/'+photo_filename, photo_filename)
        
    #Extract all images
    with zipfile.ZipFile(photo_filename, 'r') as zf:
        for member in tqdm(zf.infolist(), desc='Extracting'):
            zf.extract(member, img_folder)
      
 # Now, we need to compute the embeddings
# To speed things up, we destribute pre-computed embeddings
# Otherwise you can also encode the images yourself.
# To encode an image, you can use the following code:
# from PIL import Image
# img_emb = model.encode(Image.open(filepath))

use_precomputed_embeddings = True

if use_precomputed_embeddings: 
    emb_filename = 'unsplash-25k-photos-embeddings.pkl'
    if not os.path.exists(emb_filename):   #Download dataset if does not exist
        util.http_get('http://sbert.net/datasets/'+emb_filename, emb_filename)
        
    with open(emb_filename, 'rb') as fIn:
        img_names, img_emb = pickle.load(fIn)  
    print("Images:", len(img_names))
else:
    #For embedding images, we need the non-multilingual CLIP model
    img_model = SentenceTransformer('clip-ViT-B-32')

    img_names = list(glob.glob('photos/*.jpg'))
    print("Images:", len(img_names))
    img_emb = img_model.encode([Image.open(filepath) for filepath in img_names], batch_size=128, convert_to_tensor=True, show_progress_bar=True)

filepath = 'photos/'+img_names[0]
one_emb = torch.tensor(img_emb[0])
img_model = SentenceTransformer('clip-ViT-B-32')
comb_emb = img_model.encode(Image.open(filepath), convert_to_tensor=True).cpu()

# Next, we define a search function.
def search(query):
    # First, we encode the query (which can either be an image or a text string)
    query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    
    # Then, we use the util.semantic_search function, which computes the cosine-similarity
    # between the query embedding and all image embeddings.
    # It then returns the top_k highest ranked images, which we output
    hits = util.semantic_search(query_emb, img_emb, top_k=1)[0]
    
    print("Query:")
    display(query)
    for hit in hits:
        # print(img_names[hit['corpus_id']])
        display(IPImage(os.path.join(img_folder, img_names[hit['corpus_id']]), width=200))
        # print(os.path.join(img_folder, img_names[hit['corpus_id']]))
        return os.path.join(img_folder, img_names[hit['corpus_id']])

title = "Multilingual Joint Image & Text Embeddings"
description = "demo for Multilingual Joint Image & Text Embeddings using Sentence Transformers. To use it, simply add your text, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://www.sbert.net/'>SentenceTransformers Documentation</a> | <a href='https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/image-search'>Github Repo</a></p>"

gr.Interface(
    search, 
    gr.inputs.Textbox(label="Input"), 
    gr.outputs.Image(type="file", label="Output"),
    title=title,
    description=description,
    article=article
    ).launch()
