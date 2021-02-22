import json
import gzip
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import tqdm

paragraphs = set()

with gzip.open('wiki-programmming-20210101.jsonl.gz', 'rt') as fIn:
    for line in fIn:
        data = json.loads(line.strip())

        for p in data['paragraphs']:
            paragraphs.add(p)

paragraphs = list(paragraphs)
print("Paragraphs:", len(paragraphs))



tokenizer = T5Tokenizer.from_pretrained('BeIR/query-gen-msmarco-t5-large')
model = T5ForConditionalGeneration.from_pretrained('BeIR/query-gen-msmarco-t5-large')
model.eval()

#Select the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Parameters for generation
batch_size = 8 #Batch size
num_queries = 3 #Number of queries to generate for every paragraph
max_length_paragraph = 300 #Max length for paragraph
max_length_query = 64   #Max length for output query

with open('generated_queries.tsv', 'w') as fOut:
    for start_idx in tqdm.trange(0, len(paragraphs), batch_size):
        sub_paragraphs = paragraphs[start_idx:start_idx+batch_size]
        inputs = tokenizer.prepare_seq2seq_batch(sub_paragraphs, max_length=max_length_paragraph, truncation=True, return_tensors='pt').to(device)
        outputs = model.generate(
            **inputs,
            max_length=max_length_query,
            do_sample=True,
            top_p=0.95,
            num_return_sequences=num_queries)

        for idx, out in enumerate(outputs):
            query = tokenizer.decode(out, skip_special_tokens=True)
            para = sub_paragraphs[int(idx/num_queries)]
            fOut.write("{}\t{}\n".format(query.replace("\t", " ").strip(), para.replace("\t", " ").strip()))

