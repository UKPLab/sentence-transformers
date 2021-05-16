"""
In this example we train a semantic search model to search through Wikipedia
articles about programming articles & technologies.

We use the text paragraphs from the following Wikipedia articles:
Assembly language, C , C Sharp , C++, Go , Java , JavaScript, Keras, Laravel, MATLAB, Matplotlib, MongoDB, MySQL, Natural Language Toolkit, NumPy, pandas (software), Perl, PHP, PostgreSQL, Python , PyTorch, R , React, Rust , Scala , scikit-learn, SciPy, Swift , TensorFlow, Vue.js

In:
1_programming_query_generation.py - We generate queries for all paragraphs from these articles
2_programming_train_bi-encoder.py - We train a SentenceTransformer bi-encoder with these generated queries. This results in a model we can then use for sematic search (for the given Wikipedia articles).
3_programming_semantic_search.py - Shows how the trained model can be used for semantic search
"""
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets
import os


train_examples = []
with open('generated_queries.tsv') as fIn:
    for line in fIn:
        query, paragraph = line.strip().split('\t', maxsplit=1)
        train_examples.append(InputExample(texts=[query, paragraph]))

# For the MultipleNegativesRankingLoss, it is important
# that the batch does not contain duplicate entries, i.e.
# no two equal queries and no two equal paragraphs.
# To ensure this, we use a special data loader
train_dataloader = datasets.NoDuplicatesDataLoader(train_examples, batch_size=64)

# Now we create a SentenceTransformer model from scratch
word_emb = models.Transformer('distilbert-base-uncased')
pooling = models.Pooling(word_emb.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_emb, pooling])

# MultipleNegativesRankingLoss requires input pairs (query, relevant_passage)
# and trains the model so that is is suitable for semantic search
train_loss = losses.MultipleNegativesRankingLoss(model)


#Tune the model
num_epochs = 3
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=warmup_steps, show_progress_bar=True)

os.makedirs('output', exist_ok=True)
model.save('output/programming-model')