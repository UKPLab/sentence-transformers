import logging
import sys

from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.sparse_encoder import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator
from sentence_transformers.sparse_encoder.models import CSRSparsity

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    try:
        # Initialize model components
        model_name = "sentence-transformers/all-mpnet-base-v2"
        transformer = Transformer(model_name)
        embedding_dim = transformer.get_word_embedding_dimension()

        # Create pooling layer
        pooling = Pooling(embedding_dim, pooling_mode="mean")

        # Create sparsity layer
        csr_sparsity = CSRSparsity(
            input_dim=embedding_dim,
            hidden_dim=4 * embedding_dim,
            k=32,  # Number of top values to keep
            k_aux=512,  # Number of top values for auxiliary loss
        )

        # Create the SparseEncoder model
        model = SparseEncoder(modules=[transformer, pooling, csr_sparsity])

        # Create evaluator for all NanoBEIR datasets
        evaluator = SparseNanoBEIREvaluator(
            dataset_names=None,  # None means evaluate on all datasets
            show_progress_bar=True,
            batch_size=32,
        )

        # Run evaluation
        logger.info("Starting evaluation on all NanoBEIR datasets")
        results = evaluator(model)

        logger.info(f"Primary metric: {evaluator.primary_metric}")
        logger.info(f"Primary metric value: {results[evaluator.primary_metric]:.4f}")

        # Print results for each dataset
        for key, value in results.items():
            if key.startswith("Nano"):
                logger.info(f"{key}: {value:.4f}")

        logger.info("Evaluation completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
