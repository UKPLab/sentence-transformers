from datasets import Dataset

from sentence_transformers.sparse_encoder import (
    MLMTransformer,
    SparseEncoder,
    SparseEncoderTrainer,
    SparseMarginMSELoss,
    SpladeLoss,
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

# Create a small toy dataset
train_dataset = Dataset.from_dict(
    {
        "query": ["It's nice weather outside today.", "He drove to work."],
        "passage1": ["It's so sunny.", "He took the car to work."],
        "passage2": ["It's very sunny.", "She walked to the store."],
    }
)


def compute_labels(batch):
    emb_queries = teacher_model.encode(batch["query"])
    emb_passages1 = teacher_model.encode(batch["passage1"])
    emb_passages2 = teacher_model.encode(batch["passage2"])
    return {
        "label": teacher_model.similarity_pairwise(emb_queries, emb_passages1)
        - teacher_model.similarity_pairwise(emb_queries, emb_passages2)
    }


train_dataset = train_dataset.map(compute_labels, batched=True)
loss = SpladeLoss(
    student_model,
    main_loss=SparseMarginMSELoss,
    lamda_corpus=5e-3,
    lamda_query=0.1,
)

trainer = SparseEncoderTrainer(
    model=student_model,
    train_dataset=train_dataset,
    loss=loss,
)
trainer.train()
