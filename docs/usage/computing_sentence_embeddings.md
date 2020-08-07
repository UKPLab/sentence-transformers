# Computing Sentence Embeddings



The basic function to compute sentence embeddings looks like this:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

#Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
```


First, we load a sentence-transformer model:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('model_name_or_path')
```

You can either specify a [pre-trained model](../pretrained-models) or you can pass a path on your disc to load the sentence-transformer model from that folder.

If available, the model is automatically executed on the GPU. You can specify the device for the model like this:
```python
model = SentenceTransformer('model_name_or_path', device='cuda')
```

With *device* any pytorch device (like CPU, cuda, cuda:0 etc.)
 

The relevant method to encode a set of sentences / texts is `model.encode()`. In the following, you can find parameters this method accepts. Some relevant parameters are *batch_size* (depending on your GPU a different batch size is optimal) as well as *convert_to_numpy* (returns a numpy matrix)  and *convert_to_tensor* (returns a pytorch tensor).

```eval_rst
.. autoclass:: sentence_transformers.SentenceTransformer
    :members: encode
```

## Input Sequence Length
Transformer models like BERT / RoBERTa / DistilBERT etc. the runtime and the memory requirement grows quadratic with the input length. This limits transformers to inputs of certain lengths. A common value for BERT & Co. are 512 word pieces, which corresponde to about 300-400 words (for English). Longer texts than this are truncated to the first x word pieces.

By default, the provided methods use a limit fo 128 word pieces, longer inputs will be truncated. You can get and set the maximal sequence length like this:
 
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

print("Max Sequence Length:", model.max_seq_length)

#Change the length to 200
model.max_seq_length = 200

print("Max Sequence Length:", model.max_seq_length)
```

**Note:** You cannot increase the length higher than what is maximally supported by the respective transformer model. Also note that if a model was trained on short texts, the representations for long texts might not be that good.

## Storing & Loading Embeddings
The easiest method is to use *pickle* to store pre-computed embeddings on disc and to load it from disc. This can especially be useful if you need to encode large set of sentences. 


```python
from sentence_transformers import SentenceTransformer
import pickle

model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']


embeddings = model.encode(sentences)

#Store sentences & embeddings on disc
with open('embeddings.pkl', "wb") as fOut:
    pickle.dump({'sentences': sentences, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

#Load sentences & embeddings from disc
with open('embeddings.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_sentences = stored_data['sentences']
    stored_embeddings = stored_data['embeddings']
```