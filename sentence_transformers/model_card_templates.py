import logging

from .util import fullname

__INTRO_SECTION__ = """
# Name of Model

<!--- Describe your model here -->
"""

__EVALUATION_SECTION__ = """
## Evaluation Results

<!--- Describe how your model was evaluated -->

For an automated evaluation of this model, see the *Sentence Embeddings Benchmark*: [https://seb.sbert.net](https://seb.sbert.net)
"""

__TRAINING_SECTION__ = """
## Training
The model was trained with the parameters:

{loss_functions}

Evaluation was done with the following evaluator:
{evaluator_name}

Parameters of the fit()-Method:
```
{fit_parameters}
```
"""


__MORE_INFO_SECTION__ = """

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
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

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


__FULL_MODEL_ARCHITECTURE__ = """## Full Model Architecture
```
{full_model_str}
```"""


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


def get_train_objective_info(dataloader, loss):
    try:
        if hasattr(dataloader, 'get_config_dict'):
            train_loader = dataloader.get_config_dict()
        else:
            loader_params = {}
            loader_params['batch_size'] = dataloader.batch_size if hasattr(dataloader, 'batch_size') else 'unknown'
            if hasattr(dataloader, 'sampler'):
                loader_params['sampler'] = fullname(dataloader.sampler)
            if hasattr(dataloader, 'batch_sampler'):
                loader_params['batch_sampler'] = fullname(dataloader.batch_sampler)

        dataloader_str = """**DataLoader**:\n\n`{}` of length {} with parameters:
```
{}
```""".format(fullname(dataloader), len(dataloader), loader_params)

        loss_str = "**Loss**:\n\n`{}` {}".format(fullname(loss),
 """with parameters:
  ```
  {}
  ```""".format(loss.get_config_dict()) if hasattr(loss, 'get_config_dict') else "")

        return [dataloader_str, loss_str]

    except Exception as e:
        logging.WARN("Exception when creating get_train_objective_info: {}".format(str(e)))
        return ""