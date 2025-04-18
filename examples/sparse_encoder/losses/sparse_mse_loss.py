from datasets import Dataset

from sentence_transformers.sparse_encoder import (
    MLMTransformer,
    SparseEncoder,
    SparseEncoderTrainer,
    SparseMSELoss,
    SpladePooling,
)

# Initialize the SPLADE model
student_model_name = "prithivida/Splade_PP_en_v1"
student_model = SparseEncoder(
    modules=[
        MLMTransformer(student_model_name),
        SpladePooling(pooling_strategy="max"),  # You can also use 'sum'
    ],
    device="cuda:0",
)

# Initialize the SPLADE model
teacher_model_name = "naver/splade-cocondenser-ensembledistil"
teacher_model = SparseEncoder(
    modules=[
        MLMTransformer(teacher_model_name),
        SpladePooling(pooling_strategy="max"),  # You can also use 'sum'
    ],
    device="cuda:0",
)

train_dataset = Dataset.from_dict(
    {
        "english": ["The first sentence", "The second sentence", "The third sentence", "The fourth sentence"],
        "french": ["La première phrase", "La deuxième phrase", "La troisième phrase", "La quatrième phrase"],
    }
)


def compute_labels(batch):
    return {"label": teacher_model.encode(batch["english"], convert_to_sparse_tensor=False)}


train_dataset = train_dataset.map(compute_labels, batched=True)
loss = SparseMSELoss(student_model)

trainer = SparseEncoderTrainer(model=student_model, train_dataset=train_dataset, loss=loss)
trainer.train()
