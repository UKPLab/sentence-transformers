# Hugging Face 🤗

## The Hugging Face Hub

In addition to the official [pre-trained models](https://www.sbert.net/docs/pretrained_models.html), you can find over 500 hundred `sentence-transformer` models on the [Hugging Face Hub](http://hf.co/models?library=sentence-transformers&sort=downloads) covering a range of tasks. You can filter to find particular models, such as searching for [`feature-extraction`](https://huggingface.co/models?library=sentence-transformers&pipeline_tag=feature-extraction&sort=downloads) for models for generating embeddings, or [`sentence-similarity`](https://huggingface.co/models?library=sentence-transformers&pipeline_tag=sentence-similarity&sort=downloads) for sentence similarity models.

All models on the Hugging Face Hub come with the following:
1. An automatically generated model card with a description, example code snippets, architecture overview, and more. 
2. Metadata tags that help for discoverability and contain additional information such as a usage license.
3. An interactive widget you can use to play with the model directly in the browser.
4. An Inference API that allows to make inference requests.

<img style="height:400px;display:block;margin-left:auto;margin-right:auto;" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/libraries-sentence_transformers_widget.png"/>

## Using Hugging Face models

Any pre-trained modelson the Hub can be loaded with a single line of code 

```py
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('model_name')
```

Here is an example that loads the [multi-qa-MiniLM-L6-cos-v1 model](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1) and uses it to encode sentences and then compute the distance between them for doing semantic search.

```py
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

query_embedding = model.encode('How big is London')
passage_embedding = model.encode(['London has 9,787,426 inhabitants at the 2011 census',
                                  'London is known for its finacial district'])

print("Similarity:", util.dot_score(query_embedding, passage_embedding))
```

If you want to see how to load a specific model on the Hub, you can click `Use in sentence-transformers` and you will be given a working snippet that you can load it! 

<div style="display:flex; flex-direction:column; gap: 15px; margin-bottom: 15px;">
<img style=max-height:150px;object-fit:contain;" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/libraries-sentence_transformers_snippet1.png"/>
<img style="max-height:130px;object-fit:contain" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/libraries-sentence_transformers_snippet2.png"/>
</div>

## Sharing your models

You can share your SentenceTransformers models by using the [`save_to_hub` method](https://www.sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.save_to_hub) from a trained model.

```py
from sentence_transformers import SentenceTransformer

# Load or train a model
model.save_to_hub("my_new_model")
```

This command creates a [Hugging Face repository](https://huggingface.co/docs/hub/repositories) with an automatically generated model card, an inference widget, example code snippets, and more! [Here](https://huggingface.co/osanseviero/my_new_model) is an example.

## Additional resources

* [Hugging Face Hub docs](https://huggingface.co/docs/hub/index)
* Integration with Hub [announcement](https://huggingface.co/blog/sentence-transformers-in-the-hub).
