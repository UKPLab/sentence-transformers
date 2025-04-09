import logging

from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.sparse_encoder import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator
from sentence_transformers.sparse_encoder.models import CSRSparsity

# Set up logging
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def main():
    # Initialize model components
    model_name = "sentence-transformers/all-mpnet-base-v2"
    transformer = Transformer(model_name)
    pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
    csr_sparsity = CSRSparsity(
        input_dim=transformer.get_word_embedding_dimension(),
        hidden_dim=4 * transformer.get_word_embedding_dimension(),
        k=32,  # Number of top values to keep
        k_aux=512,  # Number of top values for auxiliary loss
    )

    # Create the SparseEncoder model
    model = SparseEncoder(modules=[transformer, pooling, csr_sparsity])

    # Load & run the evaluator on a subset of datasets
    dataset_names = ["msmarco", "nfcorpus"]
    evaluator = SparseNanoBEIREvaluator(
        dataset_names=dataset_names,
        show_progress_bar=True,
        batch_size=32,
    )

    # Run evaluation
    results = evaluator(model)

    # Print primary metric
    print(f"\nPrimary metric: {evaluator.primary_metric}")
    print(f"Primary metric value: {results[evaluator.primary_metric]:.4f}")

    for dataset in dataset_names:
        print(f"\nMetrics for {dataset}:")
        for key, value in results.items():
            if key.startswith(f"Nano{dataset}"):
                print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
