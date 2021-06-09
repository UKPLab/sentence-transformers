__INTRO_SECTION__ = """
# Name of Model

<!--- Describe your model here -->

## Model Description
The model consists of the following layers:
"""

__MORE_INFO__SECTION__ = """
## Training Procedure

<!--- Describe how your model was trained -->

## Evaluation Results

<!--- Describe how your model was evaluated -->

## Citing & Authors

<!--- Describe where people can find more information -->
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
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('model_name')
embeddings = model.encode(sentences)
print(embeddings)
```
"""

def model_card_get_pooling_function(pooling_mode):
    if pooling_mode == 'max':
        return "max_pooling", """
# Max Pooling - Take the max value over time for every dimension. 
def max_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    max_over_time = torch.max(token_embeddings, 1)[0]
    return max_over_time
"""
    elif pooling_mode == 'mean':
        return "mean_pooling", """
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask
"""

    elif pooling_mode == 'cls':
        return "cls_pooling", """
def cls_pooling(model_output, attention_mask):
    return model_output[0][:,0]
"""


__TRANSFORMERS_EXAMPLE__ = """\n
## Usage (HuggingFace Transformers)

```python
from transformers import AutoTokenizer, AutoModel
import torch

{POOLING_FUNCTION}

# Sentences we want sentence embeddings for
sentences = ['This is an example sentence']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('model_name')
model = AutoModel.from_pretrained('model_name')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, max pooling.
sentence_embeddings = {POOLING_FUNCTION_NAME}(model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
print(sentence_embeddings)
```
\n
"""
