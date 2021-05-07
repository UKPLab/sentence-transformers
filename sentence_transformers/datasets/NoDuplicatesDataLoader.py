import random
import math

class NoDuplicatesDataLoader:

    def __init__(self, train_examples, batch_size):
        """
        A special data loader to be used with MultipleNegativesRankingLoss.
        The data loader ensures that there are no duplicate sentences within the same batch
        """
        self.train_examples = train_examples
        self.batch_size = batch_size
        self.data_idx = list(range(len(train_examples)))
        self.data_pointer = 0
        self.collate_fn = None
        random.shuffle(self.data_idx)

    def __iter__(self):
        for _ in range(self.__len__()):
            batch_idx = set()
            texts_in_batch = set()

            while len(batch_idx) < self.batch_size:
                idx = self.data_idx[self.data_pointer]
                example = self.train_examples[idx]

                valid_example = True
                for text in example.texts:
                    if text.strip().lower() in texts_in_batch:
                        valid_example = False
                        break

                if valid_example:
                    batch_idx.add(idx)
                    for text in example.texts:
                        texts_in_batch.add(text.strip().lower())

                self.data_pointer += 1
                if self.data_pointer >= len(self.data_idx):
                    self.data_pointer = 0
                    random.shuffle(self.data_idx)

            batch = [self.train_examples[idx] for idx in batch_idx]
            yield self.collate_fn(batch) if self.collate_fn is not None else batch

    def __len__(self):
        return math.floor(len(self.train_examples) / self.batch_size)