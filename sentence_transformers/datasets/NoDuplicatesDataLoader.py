import math
import random


class NoDuplicatesDataLoader:
    def __init__(self, train_examples, batch_size, clean=True):
        """
        A special data loader to be used with MultipleNegativesRankingLoss.
        The data loader ensures that there are no duplicate sentences within the same batch
        """
        self.batch_size = batch_size
        self.data_pointer = 0
        self.collate_fn = None
        self.train_examples = train_examples
        self.clean = clean
        random.shuffle(self.train_examples)

    def __iter__(self):
        for _ in range(self.__len__()):
            batch = []
            texts_in_batch = set()

            while len(batch) < self.batch_size:
                example = self.train_examples[self.data_pointer]

                valid_example = True
                for text in example.texts:
                    clean_text = text.strip().lower() if self.clean else text
                    if clean_text in texts_in_batch:
                        valid_example = False
                        break

                if valid_example:
                    batch.append(example)
                    for text in example.texts:
                        clean_text = text.strip().lower() if self.clean else text
                        texts_in_batch.add(clean_text)

                self.data_pointer += 1
                if self.data_pointer >= len(self.train_examples):
                    self.data_pointer = 0
                    random.shuffle(self.train_examples)

            yield self.collate_fn(batch) if self.collate_fn is not None else batch

    def __len__(self):
        return math.floor(len(self.train_examples) / self.batch_size)
