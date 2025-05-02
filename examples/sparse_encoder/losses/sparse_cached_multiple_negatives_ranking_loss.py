from datasets import Dataset

from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, losses

# Initialize the SPLADE model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
train_dataset = Dataset.from_dict(
    {
        "anchor": ["It's nice weather outside today.", "He drove to work."],
        "positive": ["It's so sunny.", "He took the car to the office."],
    }
)
loss = losses.SparseCachedMultipleNegativesRankingLoss(model, mini_batch_size=64)

trainer = SparseEncoderTrainer(
    model=model,
    train_dataset=train_dataset,
    loss=loss,
)
trainer.train()

# TODO: Investigate if it's working with a test
