# Training Overview

## Why Finetune?
Finetuning Sparse Encoder models often heavily improves the performance of the model on your use case, because each task requires a different notion of similarity. For example, given news articles: 
- "Apple launches the new iPad"
- "NVIDIA is gearing up for the next GPU generation"

Then the following use cases, we may have different notions of similarity:
- a model for **classification** of news articles as Economy, Sports, Technology, Politics, etc., should produce **similar embeddings** for these texts.
- a model for **semantic textual similarity** should produce **dissimilar embeddings** for these texts, as they have different meanings.
- a model for **semantic search** would **not need a notion for similarity** between two documents, as it should only compare queries and documents.


Also see [**Training Examples**](training/examples) for numerous training scripts for common real-world applications that you can adopt.


## Training Components
Training Sparse Encoder models involves between 3 to 5 components just like [training Sentence Transformer models](../sentence_transformer/training_overview.md):

<div class="components">
    <a href="#dataset" class="box">
        <div class="header">Dataset</div>
        Learn how to prepare the <b>data</b> for training.
    </a>
    <a href="#loss-function" class="box">
        <div class="header">Loss Function</div>
        Learn how to prepare and choose a <b>loss</b> function.
    </a>
    <a href="#training-arguments" class="box optional">
        <div class="header">Training Arguments</div>
        Learn which <b>training arguments</b> are useful.
    </a>
    <a href="#evaluator" class="box optional">
        <div class="header">Evaluator</div>
        Learn how to <b>evaluate</b> during and after training.
    </a>
    <a href="#trainer" class="box">
        <div class="header">Trainer</div>
        Learn how to start the <b>training</b> process.
    </a>
</div>
<p></p>

## Dataset
```{eval-rst}
The :class:`SparseEncoderTrainer` trains and evaluates using :class:`datasets.Dataset` (one dataset) or :class:`datasets.DatasetDict` instances (multiple datasets, see also `Multi-dataset training <#multi-dataset-training>`_). 

.. tab:: Data on 🤗 Hugging Face Hub

    If you want to load data from the `Hugging Face Datasets <https://huggingface.co/datasets>`_, then you should use :func:`datasets.load_dataset`:

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference external" href="https://huggingface.co/docs/datasets/main/en/loading#hugging-face-hub">Datasets, Loading from the Hugging Face Hub</a></li>
                <li><a class="reference external" href="https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset" title="(in datasets vmain)"><code class="xref py py-func docutils literal notranslate"><span class="pre">datasets.load_dataset()</span></code></a></li>
                <li><a class="reference external" href="https://huggingface.co/datasets/sentence-transformers/all-nli">sentence-transformers/all-nli</a></li>
            </ul>
        </div>

    ::

        from datasets import load_dataset

        train_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="train")
        eval_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="dev")

        print(train_dataset)
        """
        Dataset({
            features: ['premise', 'hypothesis', 'label'],
            num_rows: 942069
        })
        """

    Some datasets (including `sentence-transformers/all-nli <https://huggingface.co/datasets/sentence-transformers/all-nli>`_) require you to provide a "subset" alongside the dataset name. ``sentence-transformers/all-nli`` has 4 subsets, each with different data formats: `pair <https://huggingface.co/datasets/sentence-transformers/all-nli/viewer/pair>`_, `pair-class <https://huggingface.co/datasets/sentence-transformers/all-nli/viewer/pair-class>`_, `pair-score <https://huggingface.co/datasets/sentence-transformers/all-nli/viewer/pair-score>`_, `triplet <https://huggingface.co/datasets/sentence-transformers/all-nli/viewer/triplet>`_.

    .. note::

        Many Hugging Face datasets that work out of the box with Sentence Transformers have been tagged with `sentence-transformers`, allowing you to easily find them by browsing to `https://huggingface.co/datasets?other=sentence-transformers <https://huggingface.co/datasets?other=sentence-transformers>`_. We strongly recommend that you browse these datasets to find training datasets that might be useful for your tasks.

.. tab:: Local Data (CSV, JSON, Parquet, Arrow, SQL)

    If you have local data in common file-formats, then you can load this data easily using :func:`datasets.load_dataset`:

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference external" href="https://huggingface.co/docs/datasets/main/en/loading#local-and-remote-files">Datasets, Loading local files</a></li>
                <li><a class="reference external" href="https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset" title="(in datasets vmain)"><code class="xref py py-func docutils literal notranslate"><span class="pre">datasets.load_dataset()</span></code></a></li>
            </ul>
        </div>

    ::

        from datasets import load_dataset

        dataset = load_dataset("csv", data_files="my_file.csv")
    
    or::

        from datasets import load_dataset

        dataset = load_dataset("json", data_files="my_file.json")

.. tab:: Local Data that requires pre-processing

    If you have local data that requires some extra pre-processing, my recommendation is to initialize your dataset using :meth:`datasets.Dataset.from_dict` and a dictionary of lists, like so:

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference external" href="https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.from_dict" title="(in datasets vmain)"><code class="xref py py-meth docutils literal notranslate"><span class="pre">datasets.Dataset.from_dict()</span></code></a></li>
            </ul>
        </div>

    ::

        from datasets import Dataset

        anchors = []
        positives = []
        # Open a file, do preprocessing, filtering, cleaning, etc.
        # and append to the lists

        dataset = Dataset.from_dict({
            "anchor": anchors,
            "positive": positives,
        })

    Each key from the dictionary will become a column in the resulting dataset.

```

### Dataset Format

```{eval-rst}
It is important that your dataset format matches your loss function (or that you choose a loss function that matches your dataset format). Verifying whether a dataset format works with a loss function involves two steps:

1. If your loss function requires a *Label* according to the `Loss Overview <loss_overview.html>`_ table, then your dataset must have a **column named "label" or "score"**. This column is automatically taken as the label.
2. All columns not named "label" or "score" are considered *Inputs* according to the `Loss Overview <loss_overview.html>`_ table. The number of remaining columns must match the number of valid inputs for your chosen loss. The names of these columns are **irrelevant**, only the **order matters**. 

For example, given a dataset with columns ``["text1", "text2", "label"]`` where the "label" column has float similarity score between 0 and 1, we can use it with :class:`~sentence_transformers.sparse_encoder.losses.SparseCoSENTLoss`, :class:`~sentence_transformers.sparse_encoder.losses.SparseAnglELoss`, and :class:`~sentence_transformers.sparse_encoder.losses.SparseCosineSimilarityLoss` because it:

1. has a "label" column as is required for these loss functions.
2. has 2 non-label columns, exactly the amount required by these loss functions.

Be sure to re-order your dataset columns with :meth:`Dataset.select_columns <datasets.Dataset.select_columns>` if your columns are not ordered correctly. For example, if your dataset has ``["good_answer", "bad_answer", "question"]`` as columns, then this dataset can technically be used with a loss that requires (anchor, positive, negative) triplets, but the ``good_answer`` column will be taken as the anchor, ``bad_answer`` as the positive, and ``question`` as the negative.

Additionally, if your dataset has extraneous columns (e.g. sample_id, metadata, source, type), you should remove these with :meth:`Dataset.remove_columns <datasets.Dataset.remove_columns>` as they will be used as inputs otherwise. You can also use :meth:`Dataset.select_columns <datasets.Dataset.select_columns>` to keep only the desired columns.
```

## Loss Function
Loss functions quantify how well a model performs for a given batch of data, allowing an optimizer to update the model weights to produce more favourable (i.e., lower) loss values. This is the core of the training process.

Sadly, there is no single loss function that works best for all use-cases. Instead, which loss function to use greatly depends on your available data and on your target task. See [Dataset Format](#dataset-format) to learn what datasets are valid for which loss functions. Additionally, the [Loss Overview](loss_overview) will be your best friend to learn about the options.

```{eval-rst}
Most loss functions can be initialized with just the :class:`~sentence_transformers.sparse_encoder.SparseEncoder` that you're training, alongside some optional parameters, e.g.:

.. sidebar:: Documentation

    - :class:`sentence_transformers.sparse_encoder.losses.SpladeLoss`
    - :class:`sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss`
    - `Losses API Reference <../package_reference/sparse_encoder/losses.html>`_
    - `Loss Overview <loss_overview.html>`_

::

    from datasets import load_dataset
    from sentence_transformers import SparseEncoder
    from sentence_transformers.sparse_encoder.losses import SpladeLoss, SparseMultipleNegativesRankingLoss

    # Load a model to train/finetune
    model = SparseEncoder("distilbert/distilbert-base-uncased")

    # Initialize the SpladeLoss with a SparseMultipleNegativesRankingLoss
    # This loss requires pairs of related texts or triplets
    loss = losses.SpladeLoss(
        model=model,
        loss=losses.SparseMultipleNegativesRankingLoss(model=model),
        lambda_query=5e-5,  # Weight for query loss
        lambda_corpus=3e-5,
    ) 

    # Load an example training dataset that works with our loss function:
    train_dataset = load_dataset("sentence-transformers/natural-questions", split="train")
    print(train_dataset)
    """
    Dataset({
        features: ['query', 'answer'],
        num_rows: 100231
    })
    """

```

## Training Arguments

```{eval-rst}
The :class:`~sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments` class can be used to specify parameters for influencing training performance as well as defining the tracking/debugging parameters. Although it is optional, it is heavily recommended to experiment with the various useful arguments.
```

<div class="training-arguments">
    <div class="header">Key Training Arguments for improving training performance</div>
    <div class="table">
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.learning_rate"><code>learning_rate</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.lr_scheduler_type"><code>lr_scheduler_type</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.warmup_ratio"><code>warmup_ratio</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.num_train_epochs"><code>num_train_epochs</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.max_steps"><code>max_steps</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.per_device_train_batch_size"><code>per_device_train_batch_size</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.per_device_eval_batch_size"><code>per_device_eval_batch_size</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.auto_find_batch_size "><code>auto_find_batch_size</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.fp16"><code>fp16</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.bf16"><code>bf16</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.load_best_model_at_end"><code>load_best_model_at_end</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.metric_for_best_model"><code>metric_for_best_model</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.gradient_accumulation_steps"><code>gradient_accumulation_steps</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.gradient_checkpointing"><code>gradient_checkpointing</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.eval_accumulation_steps"><code>eval_accumulation_steps</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.optim"><code>optim</code></a>
        <a href="../package_reference/sentence_transformer/training_args.html#sentence_transformers.training_args.SentenceTransformerTrainingArguments"><code>batch_sampler</code></a>
        <a href="../package_reference/sentence_transformer/training_args.html#sentence_transformers.training_args.SentenceTransformerTrainingArguments"><code>multi_dataset_batch_sampler</code></a>
        <a href="../package_reference/sentence_transformer/training_args.html#sentence_transformers.training_args.SentenceTransformerTrainingArguments"><code>prompts</code></a>
    </div>
</div>
<br>
<div class="training-arguments">
    <div class="header">Key Training Arguments for observing training performance</div>
    <div class="table">
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.eval_strategy"><code>eval_strategy</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.eval_steps"><code>eval_steps</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.save_strategy"><code>save_strategy</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.save_steps"><code>save_steps</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.save_total_limit"><code>save_total_limit</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.report_to"><code>report_to</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.run_name"><code>run_name</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.log_level"><code>log_level</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.logging_steps"><code>logging_steps</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.push_to_hub"><code>push_to_hub</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.hub_model_id"><code>hub_model_id</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.hub_strategy"><code>hub_strategy</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.hub_private_repo"><code>hub_private_repo</code></a>
    </div>
</div>
<br>

```{eval-rst}
Here is an example of how :class:`~sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments` can be initialized:
```

```python
args = SparseEncoderTrainingArguments(
    # Required parameter:
    output_dir="models/splade-distilbert-base-uncased-nq",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="splade-distilbert-base-uncased-nq",  # Will be used in W&B if `wandb` is installed
)
```

## Evaluator

You can provide the [`SparseEncoderTrainer`](https://sbert.net/docs/package_reference/sparse_encoder/trainer.html#sparse_encoder.trainer.SparseEncoderTrainer) with an `eval_dataset` to get the evaluation loss during training, but it may be useful to get more concrete metrics during training, too. For this, you can use evaluators to assess the model's performance with useful metrics before, during, or after training. You can use both an `eval_dataset` and an evaluator, one or the other, or neither. They evaluate based on the `eval_strategy` and `eval_steps` [Training Arguments](#training-arguments).

Here are the implemented Evaluators that come with Sentence Transformers for Sparse Encoder models:
```{eval-rst}
========================================================================  ===========================================================================================================================
Evaluator                                                                 Required Data
========================================================================  ===========================================================================================================================
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseBinaryClassificationEvaluator`  Pairs with class labels.
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseEmbeddingSimilarityEvaluator`   Pairs with similarity scores.
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseInformationRetrievalEvaluator`  Queries (qid => question), Corpus (cid => document), and relevant documents (qid => set[cid]).
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseNanoBEIREvaluator`              No data required.
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseMSEEvaluator`                   Source sentences to embed with a teacher model and target sentences to embed with the student model. Can be the same texts.
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseRerankingEvaluator`             List of ``{'query': '...', 'positive': [...], 'negative': [...]}`` dictionaries.
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseTranslationEvaluator`           Pairs of sentences in two separate languages.
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseTripletEvaluator`               (anchor, positive, negative) pairs.
========================================================================  ===========================================================================================================================

Additionally, :class:`~sentence_transformers.evaluation.SequentialEvaluator` should be used to combine multiple evaluators into one Evaluator that can be passed to the :class:`~sentence_transformers.sparse_encoder.trainer.SparseEncoderTrainer`.

Sometimes you don't have the required evaluation data to prepare one of these evaluators on your own, but you still want to track how well the model performs on some common benchmarks. In that case, you can use these evaluators with data from Hugging Face.

.. tab:: SparseNanoBEIREvaluator

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference internal" href="../package_reference/sparse_encoder/evaluation.html#sentence_transformers.sparse_encoder.evaluation.SparseNanoBEIREvaluator" title="sentence_transformers.sparse_encoder.evaluation.SparseNanoBEIREvaluator"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.sparse_encoder.evaluation.SparseNanoBEIREvaluator</span></code></a></li>
            </ul>
        </div>

    ::

        from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

        # Initialize the evaluator. Unlike most other evaluators, this one loads the relevant datasets
        # directly from Hugging Face, so there's no mandatory arguments
        dev_evaluator = SparseNanoBEIREvaluator()
        # You can run evaluation like so:
        # results = dev_evaluator(model)

.. tab:: SparseEmbeddingSimilarityEvaluator with STSb

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference external" href="https://huggingface.co/datasets/sentence-transformers/stsb">sentence-transformers/stsb</a></li>
                <li><a class="reference internal" href="../package_reference/sparse_encoder/evaluation.html#sentence_transformers.sparse_encoder.evaluation.SparseEmbeddingSimilarityEvaluator" title="sentence_transformers.sparse_encoder.evaluation.SparseEmbeddingSimilarityEvaluator"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.sparse_encoder.evaluation.SparseEmbeddingSimilarityEvaluator</span></code></a></li>
                <li><a class="reference internal" href="../package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SimilarityFunction" title="sentence_transformers.SimilarityFunction"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.SimilarityFunction</span></code></a></li>
            </ul>
        </div>

    ::

        from datasets import load_dataset
        from sentence_transformers.evaluation import SimilarityFunction
        from sentence_transformers.sparse_encoder.evaluation import EmbeddingSimilarityEvaluator

        # Load the STSB dataset (https://huggingface.co/datasets/sentence-transformers/stsb)
        eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")

        # Initialize the evaluator
        dev_evaluator = EmbeddingSimilarityEvaluator(
            sentences1=eval_dataset["sentence1"],
            sentences2=eval_dataset["sentence2"],
            scores=eval_dataset["score"],
            main_similarity=SimilarityFunction.COSINE,
            name="sts-dev",
        )
        # You can run evaluation like so:
        # results = dev_evaluator(model)

.. tab:: SparseTripletEvaluator with AllNLI

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference external" href="https://huggingface.co/datasets/sentence-transformers/all-nli">sentence-transformers/all-nli</a></li>
                <li><a class="reference internal" href="../package_reference/sparse_encoder/evaluation.html#sentence_transformers.sparse_encoder.evaluation.SparseTripletEvaluator" title="sentence_transformers.sparse_encoder.evaluation.SparseTripletEvaluator"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.sparse_encoder.evaluation.SparseTripletEvaluator</span></code></a></li>
                <li><a class="reference internal" href="../package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SimilarityFunction" title="sentence_transformers.SimilarityFunction"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.SimilarityFunction</span></code></a></li>
            </ul>
        </div>

    ::

        from datasets import load_dataset
        from sentence_transformers.evaluation import SimilarityFunction
        from sentence_transformers.sparse_encoder.evaluation import SparseTripletEvaluator

        # Load triplets from the AllNLI dataset (https://huggingface.co/datasets/sentence-transformers/all-nli)
        max_samples = 1000
        eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split=f"dev[:{max_samples}]")

        # Initialize the evaluator
        dev_evaluator = SparseTripletEvaluator(
            anchors=eval_dataset["anchor"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
            main_distance_function=SimilarityFunction.DOT,
            name="all-nli-dev",
        )
        # You can run evaluation like so:
        # results = dev_evaluator(model)

.. tip::

    When evaluating frequently during training with a small ``eval_steps``, consider using a tiny ``eval_dataset`` to minimize evaluation overhead. If you're concerned about the evaluation set size, a 90-1-9 train-eval-test split can provide a balance, reserving a reasonably sized test set for final evaluations. After training, you can assess your model's performance using ``trainer.evaluate(test_dataset)`` for test loss or initialize a testing evaluator with ``test_evaluator(model)`` for detailed test metrics.

    If you evaluate after training, but before saving the model, your automatically generated model card will still include the test results.

.. warning::

    When using `Distributed Training <training/distributed.html>`_, the evaluator only runs on the first device, unlike the training and evaluation datasets, which are shared across all devices. 
```

## Trainer 

```{eval-rst}
The :class:`~sentence_transformers.sparse_encoder.trainer.SparseEncoderTrainer` is where all previous components come together. We only have to specify the trainer with the model, training arguments (optional), training dataset, evaluation dataset (optional), loss function, evaluator (optional) and we can start training. Let's have a look at a script where all of these components come together:

.. sidebar:: Documentation

    #. :class:`~sentence_transformers.sparse_encoder.SparseEncoder`
    #. :class:`~sentence_transformers.sparse_encoder.model_card.SparseEncoderModelCardData`
    #. :func:`~datasets.load_dataset`
    #. :class:`~sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss`
    #. :class:`~sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments`
    #. :class:`~sentence_transformers.sparse_encoder.evaluation.SparseTripletEvaluator`
    #. :class:`~sentence_transformers.sparse_encoder.trainer.SparseEncoderTrainer`
    #. :class:`SparseEncoder.save_pretrained <sentence_transformers.sparse_encoder.SparseEncoder.save_pretrained>`
    #. :class:`SparseEncoder.push_to_hub <sentence_transformers.sparse_encoder.SparseEncoder.push_to_hub>`

    - `Training Examples <training/examples.html>`_

::

    from datasets import load_dataset
    from sentence_transformers import (
        SparseEncoder,
        SparseEncoderTrainer,
        SparseEncoderTrainingArguments,
        SparseEncoderModelCardData,
    )
    from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers
    from sentence_transformers.sparse_encoder.evaluation import SparseTripletEvaluator

    # 1. Load a model to finetune with 2. (Optional) model card data
    model = SentenceTransformer(
        "distilbert/distilbert-base-uncased",
        model_card_data=SparseEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name="Distilbert base trained on AllNLI triplets",
        )
    )

    # 3. Load a dataset to finetune on
    dataset = load_dataset("sentence-transformers/all-nli", "triplet")
    train_dataset = dataset["train"].select(range(100_000))
    eval_dataset = dataset["dev"]
    test_dataset = dataset["test"]

    # 4. Define a loss function
    loss = losses.SpladeLoss(
        model=model,
        loss=losses.SparseMultipleNegativesRankingLoss(model=model),
        lambda_query=5e-5,
        lambda_corpus=3e-5,
    )

    # 5. (Optional) Specify training arguments
    args = SparseEncoderTrainingArguments(
        # Required parameter:
        output_dir="models/splade-distilbert-base-uncased-nq",
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        run_name="splade-distilbert-base-uncased-nq",  # Will be used in W&B if `wandb` is installed
    )

    # 6. (Optional) Create an evaluator & evaluate the base model
    dev_evaluator = SparseTripletEvaluator(
        anchors=eval_dataset["anchor"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="all-nli-dev",
    )

    # 7. Create a trainer & train
    trainer = SparseEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # (Optional) Evaluate the trained model on the test set
    test_evaluator = SparseTripletEvaluator(
        anchors=test_dataset["anchor"],
        positives=test_dataset["positive"],
        negatives=test_dataset["negative"],
        name="all-nli-test",
    )
    test_evaluator(model)

    # 8. Save the trained model
    model.save_pretrained("models/splade-distilbert-base-uncased-nq/final")
    
    # 9. (Optional) Push it to the Hugging Face Hub
    model.push_to_hub("splade-distilbert-base-uncased-nq")

```

### Callbacks

```{eval-rst}
This Sparse Encoder trainer integrates support for various :class:`transformers.TrainerCallback` subclasses, such as:

- :class:`~transformers.integrations.WandbCallback` to automatically log training metrics to W&B if ``wandb`` is installed
- :class:`~transformers.integrations.TensorBoardCallback` to log training metrics to TensorBoard if ``tensorboard`` is accessible.
- :class:`~transformers.integrations.CodeCarbonCallback` to track the carbon emissions of your model during training if ``codecarbon`` is installed.

    - Note: These carbon emissions will be included in your automatically generated model card.

See the Transformers `Callbacks <https://huggingface.co/docs/transformers/main/en/main_classes/callback>`_
documentation for more information on the integrated callbacks and how to write your own callbacks.

It also integrates support to custom Callbacks, such as the one use in trainings of Splade models :class:`~sentence_transformers.sparse_encoder.callbacks.splade_callbacks.SpladeLambdaSchedulerCallback`

```

## Multi-Dataset Training

```{eval-rst}
The top performing models are trained using many datasets at once. Normally, this is rather tricky, as each dataset has a different format. However, :class:`~sentence_transformers.sparse_encoder.trainer.SparseEncoderTrainer` can train with multiple datasets without having to convert each dataset to the same format. It can even apply different loss functions to each of the datasets. The steps to train with multiple datasets are:

- Use a dictionary of :class:`~datasets.Dataset` instances (or a :class:`~datasets.DatasetDict`) as the ``train_dataset`` (and optionally also ``eval_dataset``).
- (Optional) Use a dictionary of loss functions mapping dataset names to losses. Only required if you wish to use different loss function for different datasets.

Each training/evaluation batch will only contain samples from one of the datasets. The order in which batches are samples from the multiple datasets is defined by the :class:`~sentence_transformers.training_args.MultiDatasetBatchSamplers` enum, which can be passed to the :class:`~sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments` via ``multi_dataset_batch_sampler``. Valid options are:

- ``MultiDatasetBatchSamplers.ROUND_ROBIN``: Round-robin sampling from each dataset until one is exhausted. With this strategy, it’s likely that not all samples from each dataset are used, but each dataset is sampled from equally.
- ``MultiDatasetBatchSamplers.PROPORTIONAL`` (default): Sample from each dataset in proportion to its size. With this strategy, all samples from each dataset are used and larger datasets are sampled from more frequently.
```

## Training Tips | TODO

```{eval-rst}
```

## Best Base Embedding Models | TODO

The quality of your text embedding model depends on which transformer model you choose. Sadly we cannot infer from a better performance on e.g. the GLUE or SuperGLUE benchmark that this model will also yield better representations.

To test the suitability of transformer models, I use the [training_nli_v2.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/sentence_transformer/training/nli/training_nli_v2.py) script and train on 560k (anchor, positive, negative)-triplets for 1 epoch with batch size 64. I then evaluate on 14 diverse text similarity tasks (clustering, semantic search, duplicate detection etc.) from various domains.

In the following table you find the performance for different models and their performance on this benchmark:

| Model                                                                                                                             | Performance (14 sentence similarity tasks) |
|-----------------------------------------------------------------------------------------------------------------------------------|-:-:----------------------------------------|
| [microsoft/mpnet-base](https://huggingface.co/microsoft/mpnet-base)                                                               | 60.99                                      |
| [nghuyong/ernie-2.0-en](https://huggingface.co/nghuyong/ernie-2.0-en)                                                             | 60.73                                      |
| [microsoft/deberta-base](https://huggingface.co/microsoft/deberta-base)                                                           | 60.21                                      |
| [roberta-base](https://huggingface.co/roberta-base)                                                                               | 59.63                                      |
| [t5-base](https://huggingface.co/t5-base)                                                                                         | 59.21                                      |
| [bert-base-uncased](https://huggingface.co/bert-base-uncased)                                                                     | 59.17                                      |
| [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)                                                         | 59.03                                      |
| [nreimers/TinyBERT_L-6_H-768_v2](https://huggingface.co/nreimers/TinyBERT_L-6_H-768_v2)                                           | 58.27                                      |
| [google/t5-v1_1-base](https://huggingface.co/google/t5-v1_1-base)                                                                 | 57.63                                      |
| [nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Large](https://huggingface.co/nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Large) | 57.31                                      |
| [albert-base-v2](https://huggingface.co/albert-base-v2)                                                                           | 57.14                                      |
| [microsoft/MiniLM-L12-H384-uncased](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased)                                     | 56.79                                      |
| [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base)                                                     | 54.46                                      |

## Comparisons with SentenceTransformer Training | TODO

```{eval-rst}
Training :class:`~sentence_transformers.sparse_encoder.SparseEncoder` models is very similar as training :class:`~sentence_transformers.SentenceTransformer` models. 

See the `Sentence Transformer > Training Overview <../sentence_transformer/training_overview.html>`_ documentation for more details on training :class:`~sentence_transformers.SentenceTransformer` models.

```