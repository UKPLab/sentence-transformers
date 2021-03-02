"""
A quantized model executes some or all of the operations with integers rather than floating point values. This allows for a more compact models and the use of high performance vectorized operations on many hardware platforms.

As a result, you get about 40% smaller and faster models. The speed-up depends on your CPU and how PyTorch was build and can be anywhere between 10% speed-up and 300% speed-up.

Note: Quantized models are only available for CPUs. Use a GPU, if available, for optimal performance.

For more details:
https://pytorch.org/docs/stable/quantization.html
"""
import logging
import os
import torch
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.nn import Embedding, Linear
from torch.quantization import quantize_dynamic
import gzip
import csv
import time

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


#Check if dataset exsist. If not, download and extract  it
sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

#Limit torch to 4 threads
torch.set_num_threads(4)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
### /print debug information to stdout

model_name = 'paraphrase-distilroberta-base-v1'

# Load a named sentence model (based on BERT). This will download the model from our server.
# Alternatively, you can also pass a filepath to SentenceTransformer()
model = SentenceTransformer(model_name, device='cpu')
q_model = quantize_dynamic(model, {Linear, Embedding})


# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark dataset")
test_samples = []
sentences = []

with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

        sentences.append(row['sentence1'])
        sentences.append(row['sentence2'])

        if row['split'] == 'test':
            test_samples.append(inp_example)

sentences = sentences[0:10000]

logging.info("Evaluating speed of unquantized model")
start_time = time.time()
emb = model.encode(sentences, show_progress_bar=True)
diff_normal = time.time() - start_time
logging.info("Done after {:.2f} sec. {:.2f} sentences / sec".format(diff_normal, len(sentences) / diff_normal))

logging.info("Evaluating speed of quantized model")
start_time = time.time()
emb = q_model.encode(sentences, show_progress_bar=True)
diff_quantized = time.time() - start_time
logging.info("Done after {:.2f} sec. {:.2f} sentences / sec".format(diff_quantized, len(sentences) / diff_quantized))
logging.info("Speed-up: {:.2f}".format(diff_normal / diff_quantized))
#########

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')

logging.info("Evaluate regular model")
model.evaluate(evaluator)

print("\n\n")
logging.info("Evaluate quantized model")
q_model.evaluate(evaluator)





