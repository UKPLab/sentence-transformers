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

**Note:** Even though we talk about sentence embeddings, you can use it also for shorter phrases as well as for longer texts with multiple sentences. See the section on Input Sequence Length for more notes on embeddings for paragraphs.

First, we load a sentence-transformer model:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('model_name_or_path')
```

You can either specify a [pre-trained model](https://www.sbert.net/docs/pretrained_models.html) or you can pass a path on your disc to load the sentence-transformer model from that folder.

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

## Multi-Process / Multi-GPU Encoding

You can encode input texts with more than one GPU (or with multiple processes on a CPU machine). For an example, see: [computing_embeddings_mutli_gpu.py](computing_embeddings_mutli_gpu.py).

The relevant method is `start_multi_process_pool()`, which starts multiple processes that are used for encoding.

 ```eval_rst
.. automethod:: sentence_transformers.SentenceTransformer.start_multi_process_pool
```

## Sentence Embeddings with Transformers
Most of our pre-trained models are based on [Huggingface.co/Transformers](https://huggingface.co/transformers/) and are also hosted in the [models repository](https://huggingface.co/models) from Huggingface. It is possible to use our sentence embeddings models without installing sentence-transformers:

```python
from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask



#Sentences we want sentence embeddings for
sentences = ['This framework generates embeddings for each input sentence',
             'Sentences are passed as a list of string.',
             'The quick brown fox jumps over the lazy dog.']

#Load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")

#Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

#Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

#Perform pooling. In this case, mean pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
```


You can find the available models here: [https://huggingface.co/sentence-transformers](https://huggingface.co/sentence-transformers)


In the above example we add mean pooling on top of the AutoModel (which will load a BERT model). We also have models with max-pooling and where we use the CLS token. How to apply this pooling correctly, have a look at [sentence-transformers/bert-base-nli-max-tokens](https://huggingface.co/sentence-transformers/bert-base-nli-max-tokens) and [/sentence-transformers/bert-base-nli-cls-token](https://huggingface.co/sentence-transformers/bert-base-nli-cls-token).


