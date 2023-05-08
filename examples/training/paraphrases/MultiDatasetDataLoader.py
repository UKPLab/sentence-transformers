import math
import logging
import random

class MultiDatasetDataLoader:
    def __init__(self, datasets, batch_size_pairs, batch_size_triplets=None, dataset_size_temp=-1):
        self.allow_swap = True
        self.batch_size_pairs = batch_size_pairs
        self.batch_size_triplets = batch_size_pairs if batch_size_triplets is None else batch_size_triplets

        # Compute dataset weights
        self.dataset_lengths = list(map(len, datasets))
        self.dataset_lengths_sum = sum(self.dataset_lengths)

        weights = []
        if dataset_size_temp > 0:  # Scale probability with dataset size
            for dataset in datasets:
                prob = len(dataset) / self.dataset_lengths_sum
                weights.append(max(1, int(math.pow(prob, 1 / dataset_size_temp) * 1000)))
        else:  # Equal weighting of all datasets
            weights = [100] * len(datasets)

        logging.info("Dataset lenghts and weights: {}".format(list(zip(self.dataset_lengths, weights))))

        self.dataset_idx = []
        self.dataset_idx_pointer = 0

        for idx, weight in enumerate(weights):
            self.dataset_idx.extend([idx] * weight)
        random.shuffle(self.dataset_idx)

        self.datasets = []
        for dataset in datasets:
            random.shuffle(dataset)
            self.datasets.append({
                'elements': dataset,
                'pointer': 0,
            })

    def __iter__(self):
        for _ in range(int(self.__len__())):
            # Select dataset
            if self.dataset_idx_pointer >= len(self.dataset_idx):
                self.dataset_idx_pointer = 0
                random.shuffle(self.dataset_idx)

            dataset_idx = self.dataset_idx[self.dataset_idx_pointer]
            self.dataset_idx_pointer += 1

            # Select batch from this dataset
            dataset = self.datasets[dataset_idx]
            batch_size = self.batch_size_pairs if len(dataset['elements'][0].texts) == 2 else self.batch_size_triplets

            batch = []
            texts_in_batch = set()
            guid_in_batch = set()
            while len(batch) < batch_size:
                example = dataset['elements'][dataset['pointer']]

                valid_example = True
                # First check if one of the texts in already in the batch
                for text in example.texts:
                    text_norm = text.strip().lower()
                    if text_norm in texts_in_batch:
                        valid_example = False

                    texts_in_batch.add(text_norm)

                # If the example has a guid, check if guid is in batch
                if example.guid is not None:
                    valid_example = valid_example and example.guid not in guid_in_batch
                    guid_in_batch.add(example.guid)


                if valid_example:
                    if self.allow_swap and random.random() > 0.5:
                        example.texts[0], example.texts[1] = example.texts[1], example.texts[0]

                    batch.append(example)

                dataset['pointer'] += 1
                if dataset['pointer'] >= len(dataset['elements']):
                    dataset['pointer'] = 0
                    random.shuffle(dataset['elements'])

            yield self.collate_fn(batch) if self.collate_fn is not None else batch

    def __len__(self):
        return int(self.dataset_lengths_sum / self.batch_size_pairs)
