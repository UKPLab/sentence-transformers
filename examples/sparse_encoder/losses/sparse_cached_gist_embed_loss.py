from datasets import Dataset

from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, losses

model = SparseEncoder("sparse-embedding/splade-distilbert-base-uncased-init")
guide = SparseEncoder("naver/splade-cocondenser-ensembledistil")
train_dataset = Dataset.from_dict(
    {
        "anchor": ["It's nice weather outside today.", "He drove to work."],
        "positive": ["It's so sunny.", "He took the car to the office."],
    }
)
loss = losses.SparseCachedGISTEmbedLoss(
    model,
    guide,
    mini_batch_size=64,
    margin_strategy="relative",  # or "relative" (e.g., margin=0.05 for max. 95% of positive similarity)
    margin=0.1,
)

trainer = SparseEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)
trainer.train()

# TODO: Investigate if it's working with a test, seems that the problem is hparam and not the cache part
