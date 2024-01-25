from transformers import TrainingArguments as TransformersTrainingArguments


class TrainingArguments(TransformersTrainingArguments):
    # TODO: This may be required for e.g. round_robin=False, which would be set to True for
    # backwards compatibility for SentenceTransformer.fit
    pass
