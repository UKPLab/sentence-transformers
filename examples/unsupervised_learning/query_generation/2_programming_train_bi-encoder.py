from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
import random
import os
import math

train_examples = []
with open('generated_queries.tsv') as fIn:
    for line in fIn:
        query, paragraph = line.strip().split('\t', maxsplit=1)
        train_examples.append(InputExample(texts=[query, paragraph]))


class NoDuplicatesSampler:
    def __init__(self, train_examples, batch_size):
        self.train_examples = train_examples
        self.batch_size = batch_size
        self.data_idx = list(range(len(train_examples)))
        self.data_pointer = 0
        random.shuffle(self.data_idx)

    def __iter__(self):
        for _ in range(self.__len__()):
            batch_idx = set()
            texts_in_batch = set()

            while len(batch_idx) < self.batch_size:
                example = self.train_examples[self.data_idx[self.data_pointer]]

                valid_example = True
                for text in example.texts:
                    if text.strip().lower() in texts_in_batch:
                        valid_example = False
                        break

                if valid_example:
                    batch_idx.add(self.data_pointer)
                    for text in example.texts:
                        texts_in_batch.add(text.strip().lower())

                self.data_pointer += 1
                if self.data_pointer >= len(self.data_idx):
                    self.data_pointer = 0
                    random.shuffle(self.data_idx)

            yield list(batch_idx)

    def __len__(self):
        return math.ceil(len(self.train_examples) / self.batch_size)




train_dataloader = DataLoader(train_examples, batch_sampler=NoDuplicatesSampler(train_examples, batch_size=64))

word_emb = models.Transformer('distilbert-base-uncased')
pooling = models.Pooling(word_emb.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_emb, pooling])

train_loss = losses.MultipleNegativesRankingLoss(model)


#Tune the model
num_epochs = 3
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=warmup_steps, show_progress_bar=True)

os.makedirs('output', exist_ok=True)
model.save('output/programming-model')