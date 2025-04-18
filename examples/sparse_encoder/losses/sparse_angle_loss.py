from datasets import Dataset

from sentence_transformers.sparse_encoder import (
    MLMTransformer,
    SparseAnglELoss,
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
        "sentence1": ["It's nice weather outside today.", "He drove to work."],
        "sentence2": ["It's so sunny.", "She walked to the store."],
        "score": [1.0, 0.3],
    }
)

# Initialize the sparse loss
loss = SparseAnglELoss(model)

# Create the trainer
trainer = SparseEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)

# Train the model
trainer.train()
