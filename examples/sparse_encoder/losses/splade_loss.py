"""
Example showing how to use the SpladeLambdaSchedulerCallback to gradually
increase the lambda parameters during training of a SPLADE model.
"""

from datasets import Dataset

from sentence_transformers.sparse_encoder import (
    MLMTransformer,
    SchedulerType,
    SparseEncoder,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
    SparseMarginMSELoss,
    SpladeLambdaSchedulerCallback,
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
    main_loss=SparseMarginMSELoss(student_model),
    lambda_corpus=5e-3,
    lambda_query=0.1,
)

# Create the callback with explicit parameters
splade_callback = SpladeLambdaSchedulerCallback(
    loss=loss,
    scheduler_type=SchedulerType.QUADRATIC,  # Can be LINEAR or QUADRATIC
    warmup_ratio=1 / 3,  # Will reach max values after 20% of total steps
)
training_args = SparseEncoderTrainingArguments(
    num_train_epochs=20,
    per_device_train_batch_size=2,
    output_dir="runs/splade_with_lambda_scheduling",
    logging_steps=1,
)
# Create the trainer with the callback
trainer = SparseEncoderTrainer(
    model=student_model,
    train_dataset=train_dataset,
    loss=loss,
    callbacks=[splade_callback],  # Explicitly add the callback
    args=training_args,
)

# Train the model with the scheduler active
trainer.train()

# Note:
# 1. The lambda values of SpladeLoss will start at 0 and gradually increase to their maximum values
# 2. When using the QUADRATIC scheduler, the values increase more slowly at first
# 3. If you don't add the callback manually, the SparseEncoderTrainer will add it automatically
#    when it detects a SpladeLoss is being used, but with a linear scheduler and 1/3 warmup ratio
