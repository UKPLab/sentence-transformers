__INTRO_SECTION__ = """
# TODO: Name of Model

TODO: Description

## Model Description
TODO: Add relevant content
"""

__MORE_INFO__SECTION__ = """
## TODO: Training Procedure

## TODO: Evaluation Results

## TODO: Citing & Authors
"""

__SENTENCE_TRANSFORMERS_EXAMPLE__ = """
## Usage (Sentence-Transformers)

Using this model becomes more convenient when you have [sentence-transformers](https://github.com/UKPLab/sentence-transformers) installed:

```
pip install -U sentence-transformers
```

Then you can use the model like this:

```python
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence"]

model = SentenceTransformer(TODO)
embeddings = model.encode(sentences)
print(embeddings)
```
"""

__TRANSFORMERS_EXAMPLE__ = """\n
## Usage (HuggingFace Transformers)

```python
from transformers import AutoTokenizer, AutoModel
import torch

# The next step is optional if you want your own pooling function.
# Max Pooling - Take the max value over time for every dimension. 
def max_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    max_over_time = torch.max(token_embeddings, 1)[0]
    return max_over_time

# Sentences we want sentence embeddings for
sentences = ['This is an example sentence']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(TODO)
model = AutoModel.from_pretrained(TODO)

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt'))

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, max pooling.
sentence_embeddings = max_pooling(model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
print(sentence_embeddings)
```
\n
"""