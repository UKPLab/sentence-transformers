from datasets import Dataset

from sentence_transformers.sparse_encoder import (
    MLMTransformer,
    SparseEncoder,
    SparseEncoderTrainer,
    SparseTripletLoss,
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
        "negative": ["It's raining heavily.", "He walked to work."],
    }
)

# Initialize the sparse loss
loss = SparseTripletLoss(model)

# Create the trainer
trainer = SparseEncoderTrainer(
    model=model,
    train_dataset=train_dataset,
    loss=loss,
)

# Train the model
trainer.train()
