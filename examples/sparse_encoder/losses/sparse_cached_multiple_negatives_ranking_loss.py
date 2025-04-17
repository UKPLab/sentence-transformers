from datasets import Dataset

from sentence_transformers.sparse_encoder import (
    MLMTransformer,
    SparseCachedMultipleNegativesRankingLoss,
    SparseEncoder,
    SparseEncoderTrainer,
    SpladePooling,
)

# Initialize the SPLADE model
model_name = "naver/splade-cocondenser-ensembledistil"
model = SparseEncoder(
    modules=[
        MLMTransformer(model_name),
        SpladePooling(pooling_strategy="max"),  # You can also use 'sum'
    ],
    device="cuda:0",
)
# Create a small toy dataset
train_dataset = Dataset.from_dict(
    {
        "anchor": ["It's nice weather outside today.", "He drove to work."],
        "positive": ["It's so sunny.", "He took the car to the office."],
    }
)

# Initialize the sparse loss with caching
loss = SparseCachedMultipleNegativesRankingLoss(model, mini_batch_size=64)

# Create the trainer
trainer = SparseEncoderTrainer(
    model=model,
    train_dataset=train_dataset,
    loss=loss,
)

# Train the model
trainer.train()
