
# Migration Guide

## Migrating from v2.x to v3.x

```{eval-rst}
The v3 Sentence Transformers release refactored the training of :class:`~sentence_transformers.SentenceTransformer` embedding models, replacing :meth:`SentenceTransformer.fit <sentence_transformers.SentenceTransformer.fit>` with a :class:`~sentence_transformers.trainer.SentenceTransformerTrainer` and :class:`~sentence_transformers.training_args.SentenceTransformerTrainingArguments`. This update softly deprecated :meth:`SentenceTransformer.fit <sentence_transformers.SentenceTransformer.fit>`, meaning that it still works, but it's recommended to switch to the new v3.x training format. Behind the scenes, this method now uses the new trainer.

.. warning::
    If you don't have code that uses :meth:`SentenceTransformer.fit <sentence_transformers.SentenceTransformer.fit>`, then you will not have to make any changes to your code to update from v2.x to v3.x.

    If you do, your code still works, but it is recommended to switch to the new v3.x training format, as it allows more training arguments and functionality. See the `Training Overview <sentence_transformer/training_overview.html>`_ for more details.

.. list-table:: Old and new training flow
   :widths: 50 50
   :header-rows: 1

   * - v2.x
     - v3.x (recommended)
   * - ::

        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader

        # 1. Define the model. Either from scratch of by loading a pre-trained model
        model = SentenceTransformer("microsoft/mpnet-base")

        # 2. Define your train examples. You need more than just two examples...
        train_examples = [
            InputExample(texts=[
                "A person on a horse jumps over a broken down airplane.",
                "A person is outdoors, on a horse.",
                "A person is at a diner, ordering an omelette.",
            ]),
            InputExample(texts=[
                "Children smiling and waving at camera",
                "There are children present",
                "The kids are frowning",
            ]),
        ]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

        # 3. Define a loss function
        train_loss = losses.MultipleNegativesRankingLoss(model)

        # 4. Finetune the model
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=100,
        )

        # 5. Save the trained model
        model.save_pretrained("models/mpnet-base-all-nli")
     - ::

        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
        from sentence_transformers.losses import MultipleNegativesRankingLoss

        # 1. Define the model. Either from scratch of by loading a pre-trained model
        model = SentenceTransformer("microsoft/mpnet-base")

        # 2. Load a dataset to finetune on
        dataset = load_dataset("sentence-transformers/all-nli", "triplet")
        train_dataset = dataset["train"].select(range(10_000))
        eval_dataset = dataset["dev"].select(range(1_000))

        # 3. Define a loss function
        loss = MultipleNegativesRankingLoss(model)

        # 4. Create a trainer & train
        trainer = SentenceTransformerTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
        )
        trainer.train()

        # 5. Save the trained model
        model.save_pretrained("models/mpnet-base-all-nli")
        # model.push_to_hub("mpnet-base-all-nli")
```

### Migration for specific parameters from `SentenceTransformer.fit`
```{eval-rst}
.. collapse:: SentenceTransformer.fit(train_objectives)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 5-17, 20, 23
   
           from sentence_transformers import SentenceTransformer, InputExample, losses
           from torch.utils.data import DataLoader
   
           # Define a training dataloader
           train_examples = [
               InputExample(texts=[
                   "A person on a horse jumps over a broken down airplane.",
                   "A person is outdoors, on a horse.",
                   "A person is at a diner, ordering an omelette.",
               ]),
               InputExample(texts=[
                   "Children smiling and waving at camera",
                   "There are children present",
                   "The kids are frowning",
               ]),
           ]
           train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
   
           # Define a loss function
           train_loss = losses.MultipleNegativesRankingLoss(model)

           # Finetune the model
           model.fit(train_objectives=[(train_dataloader, train_loss)])
        - .. code-block:: python
           :emphasize-lines: 6-18, 21, 26, 27

           from datasets import Dataset
           from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
           from sentence_transformers.losses import MultipleNegativesRankingLoss

           # Define a training dataset
           train_examples = [
               {
                   "anchor": "A person on a horse jumps over a broken down airplane.",
                   "positive": "A person is outdoors, on a horse.",
                   "negative": "A person is at a diner, ordering an omelette.",
               },
               {
                   "anchor": "Children smiling and waving at camera",
                   "positive": "There are children present",
                   "negative": "The kids are frowning",
               },
           ]
           train_dataset = Dataset.from_list(train_examples)
   
           # Define a loss function
           loss = MultipleNegativesRankingLoss(model)
   
           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(evaluator)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 9
   
           ...
   
           # Load an evaluator
           evaluator = NanoBEIREvaluator()
   
           # Finetune with an evaluator
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               evaluator=evaluator,
           )
        - .. code-block:: python
           :emphasize-lines: 10
   
           # Load an evaluator
           evaluator = NanoBEIREvaluator()
   
           # Finetune with an evaluator
           trainer = SentenceTransformerTrainer(
               model=model,
               train_dataset=train_dataset,
               eval_dataset=eval_dataset,
               loss=loss,
               evaluator=evaluator,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(epochs)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6
   
           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               epochs=1,
           )
        - .. code-block:: python
           :emphasize-lines: 5
   
           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               num_train_epochs=1,
           )
   
           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(steps_per_epoch)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6
   
           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               steps_per_epoch=1000,
           )
        - .. code-block:: python
           :emphasize-lines: 5
   
           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               max_steps=1000, # Note: max_steps is across all epochs, not per epoch
           )
   
           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(scheduler)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6
   
           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               scheduler="WarmupLinear",
           )
        - .. code-block:: python
           :emphasize-lines: 6
   
           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               # See https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.SchedulerType
               lr_scheduler_type="linear"
           )
   
           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(warmup_steps)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6
   
           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               warmup_steps=1000,
           )
        - .. code-block:: python
           :emphasize-lines: 5
   
           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               warmup_steps=1000,
           )
   
           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(optimizer_class, optimizer_params)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6
   
           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               optimizer_class=torch.optim.AdamW,
               optimizer_params={"eps": 1e-7},
           )
        - .. code-block:: python
           :emphasize-lines: 6-7
   
           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               # See https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
               optim="adamw_torch",
               optim_args={"eps": 1e-7},
           )
   
           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(weight_decay)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6
   
           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               weight_decay=0.02,
           )
        - .. code-block:: python
           :emphasize-lines: 5
   
           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               weight_decay=0.02,
           )
   
           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(evaluation_steps)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6, 7
   
           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               evaluator=evaluator,
               evaluation_steps=1000,
           )
        - .. code-block:: python
           :emphasize-lines: 5, 6, 10, 15, 17
   
           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               eval_strategy="steps",
               eval_steps=1000,
           )
   
           # Finetune the model
           # Note: You need an eval_dataset and/or evaluator to evaluate
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               eval_dataset=eval_dataset,
               loss=loss,
               evaluator=evaluator,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(output_path, save_best_model)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 7, 8
   
           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               evaluator=evaluator,
               output_path="my/path",
               save_best_model=True,
           )
        - .. code-block:: python
           :emphasize-lines: 5, 6, 19
   
           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               load_best_model_at_end=True,
               metric_for_best_model="all_nli_cosine_accuracy", # E.g. `evaluator.primary_metric`
           )
   
           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

           # Save the best model at my output path
           model.save_pretrained("my/path")

.. collapse:: SentenceTransformer.fit(max_grad_norm)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6
   
           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               max_grad_norm=1,
           )
        - .. code-block:: python
           :emphasize-lines: 5
   
           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               max_grad_norm=1,
           )
   
           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(use_amp)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6
   
           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               use_amp=True,
           )
        - .. code-block:: python
           :emphasize-lines: 5, 6
   
           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               fp16=True,
               bf16=False, # If your GPU supports it, you can also use bf16 instead
           )
   
           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(callback)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 3, 4, 9
   
           ...

           def printer_callback(score, epoch, steps):
               print(f"Score: {score:.4f} at epoch {epoch:d}, step {steps:d}")

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               callback=printer_callback,
           )
        - .. code-block:: python
           :emphasize-lines: 1, 5-10, 17
   
           from transformers import TrainerCallback

           ...

           class PrinterCallback(TrainerCallback):
               # Subclass any method from https://huggingface.co/docs/transformers/main_classes/callback#transformers.TrainerCallback
               def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                   print(f"Metrics: {metrics} at epoch {state.epoch:d}, step {state.global_step:d}")

           printer_callback = PrinterCallback()

           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               train_dataset=train_dataset,
               loss=loss,
               callbacks=[printer_callback],
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(show_progress_bar)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6
   
           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               show_progress_bar=True,
           )
        - .. code-block:: python
           :emphasize-lines: 5
   
           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               disable_tqdm=False,
           )
   
           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(checkpoint_path, checkpoint_save_steps, checkpoint_save_total_limit)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6-8
   
           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               checkpoint_path="checkpoints",
               checkpoint_save_steps=5000,
               checkpoint_save_total_limit=2,
           )
        - .. code-block:: python
           :emphasize-lines: 7-9, 13, 18
   
           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               eval_strategy="steps",
               eval_steps=5000,
               save_strategy="steps",
               save_steps=5000,
               save_total_limit=2,
           )
   
           # Finetune the model
           # Note: You need an eval_dataset and/or evaluator to checkpoint
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               eval_dataset=eval_dataset,
               loss=loss,
           )
           trainer.train()
```

<br>

### Migration for custom Datasets and DataLoaders used in `SentenceTransformer.fit`

```{eval-rst}
.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - v2.x
     - v3.x (recommended)
   * - ``ParallelSentencesDataset``
     - Manually creating a :class:`~datasets.Dataset` and adding a ``label`` column for embeddings. Alternatively, consider loading one of our pre-provided `Parallel Sentences Datasets <https://huggingface.co/collections/sentence-transformers/parallel-sentences-datasets-6644d644123d31ba5b1c8785>`_.
   * - ``SentenceLabelDataset``
     - Loading or creating a :class:`~datasets.Dataset` and using ``SentenceTransformerTrainingArguments(batch_sampler=BatchSamplers.GROUP_BY_LABEL)`` (uses the :class:`~sentence_transformers.sampler.GroupByLabelBatchSampler`). Recommended for the BatchTripletLosses.
   * - ``DenoisingAutoEncoderDataset``
     - Manually adding a column with noisy text to a :class:`~datasets.Dataset` with texts, e.g. with :func:`Dataset.map <datasets.Dataset.map>`.
   * - ``NoDuplicatesDataLoader``
     - Loading or creating a :class:`~datasets.Dataset` and using ``SentenceTransformerTrainingArguments(batch_sampler=BatchSamplers.NO_DUPLICATES)`` (uses the :class:`~sentence_transformers.sampler.NoDuplicatesBatchSampler`). Recommended for :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss`.
```

## Migrating from v3.x to v4.x

```{eval-rst}
The v4 Sentence Transformers release refactored the training of :class:`~sentence_transformers.cross_encoder.CrossEncoder` reranker/pair classification models, replacing :meth:`CrossEncoder.fit <sentence_transformers.SentenceTransformer.fit>` with a :class:`~sentence_transformers.cross_encoder.trainer.CrossEncoderTrainer` and :class:`~sentence_transformers.cross_encoder..training_args.CrossEncoderTrainingArguments`. Like with v3 and :class:`~sentence_transformers.SentenceTransformer` models, this update softly deprecated :meth:`CrossEncoder.fit <sentence_transformers.cross_encoder.CrossEncoder.fit>`, meaning that it still works, but it's recommended to switch to the new v4.x training format. Behind the scenes, this method now uses the new trainer.

.. warning::
    If you don't have code that uses :meth:`CrossEncoder.fit <sentence_transformers.cross_encoder.CrossEncoder.fit>`, then you will not have to make any changes to your code to update from v3.x to v4.x.

    If you do, your code still works, but it is recommended to switch to the new v4.x training format, as it allows more training arguments and functionality. See the `Training Overview <cross_encoder/training_overview.html>`_ for more details.

.. list-table:: Old and new training flow
   :widths: 50 50
   :header-rows: 1

   * - v3.x
     - v4.x (recommended)
   * - ::

        from sentence_transformers import CrossEncoder, InputExample
        from torch.utils.data import DataLoader
        
        # 1. Define the model. Either from scratch of by loading a pre-trained model
        model = CrossEncoder("microsoft/mpnet-base")
        
        # 2. Define your train examples. You need more than just two examples...
        train_examples = [
            InputExample(texts=["What are pandas?", "The giant panda ..."], label=1),
            InputExample(texts=["What's a panda?", "Mount Vesuvius is a ..."], label=0),
        ]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        
        # 3. Finetune the model
        model.fit(train_dataloader=train_dataloader, epochs=1, warmup_steps=100)
     - ::

        from datasets import load_dataset
        from sentence_transformers import CrossEncoder, CrossEncoderTrainer
        from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
        
        # 1. Define the model. Either from scratch of by loading a pre-trained model
        model = CrossEncoder("microsoft/mpnet-base")
        
        # 2. Load a dataset to finetune on, convert to required format
        dataset = load_dataset("sentence-transformers/hotpotqa", "triplet", split="train")
        
        def triplet_to_labeled_pair(batch):
            anchors = batch["anchor"]
            positives = batch["positive"]
            negatives = batch["negative"]
            return {
                "sentence_A": anchors * 2,
                "sentence_B": positives + negatives,
                "labels": [1] * len(positives) + [0] * len(negatives),
            }
        
        dataset = dataset.map(triplet_to_labeled_pair, batched=True, remove_columns=dataset.column_names)
        train_dataset = dataset.select(range(10_000))
        eval_dataset = dataset.select(range(10_000, 11_000))
        
        # 3. Define a loss function
        loss = BinaryCrossEntropyLoss(model)
        
        # 4. Create a trainer & train
        trainer = CrossEncoderTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
        )
        trainer.train()
        
        # 5. Save the trained model
        model.save_pretrained("models/mpnet-base-hotpotqa")
        # model.push_to_hub("mpnet-base-hotpotqa")

```

### Migration for parameters on `CrossEncoder` initialization and methods

```{eval-rst}
.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - v3.x
     - v4.x (recommended)
   * - ``CrossEncoder(model_name=...)``
     - Renamed to ``CrossEncoder(model_name_or_path=...)``
   * - ``CrossEncoder(automodel_args=...)``
     - Renamed to ``CrossEncoder(model_kwargs=...)``
   * - ``CrossEncoder(tokenizer_args=...)``
     - Renamed to ``CrossEncoder(tokenizer_kwargs=...)``
   * - ``CrossEncoder(config_args=...)``
     - Renamed to ``CrossEncoder(config_kwargs=...)``
   * - ``CrossEncoder(cache_dir=...)``
     - Renamed to ``CrossEncoder(cache_folder=...)``
   * - ``CrossEncoder(default_activation_function=...)``
     - Renamed to ``CrossEncoder(activation_fn=...)``
   * - ``CrossEncoder(classifier_dropout=...)``
     - Use ``CrossEncoder(config_kwargs={"classifier_dropout": ...})`` instead.
   * - ``CrossEncoder.predict(activation_fct=...)``
     - Renamed to ``CrossEncoder.predict(activation_fn=...)``
   * - ``CrossEncoder.rank(activation_fct=...)``
     - Renamed to ``CrossEncoder.rank(activation_fn=...)``
   * - ``CrossEncoder.predict(num_workers=...)``
     - Fully deprecated, no longer has any effect.
   * - ``CrossEncoder.rank(num_workers=...)``
     - Fully deprecated, no longer has any effect.

.. note::

   The old keyword arguments still work, but they will emit a warning recommending you to use the new names instead.
```

### Migration for specific parameters from `CrossEncoder.fit`
```{eval-rst}
.. collapse:: CrossEncoder.fit(train_dataloader)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 8-12, 15
   
           from sentence_transformers import CrossEncoder, InputExample
           from torch.utils.data import DataLoader
           
           # 1. Define the model. Either from scratch of by loading a pre-trained model
           model = CrossEncoder("microsoft/mpnet-base")
           
           # 2. Define your train examples. You need more than just two examples...
           train_examples = [
               InputExample(texts=["What are pandas?", "The giant panda ..."], label=1),
               InputExample(texts=["What's a panda?", "Mount Vesuvius is a ..."], label=0),
           ]
           train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
           
           # 3. Finetune the model
           model.fit(train_dataloader=train_dataloader)
        - .. code-block:: python
           :emphasize-lines: 6-18, 26

           from datasets import Dataset
           from sentence_transformers import CrossEncoder, CrossEncoderTrainer
           from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

           # Define a training dataset
           train_examples = [
               {
                   "sentence_1": "A person on a horse jumps over a broken down airplane.",
                   "sentence_2": "A person is outdoors, on a horse.",
                   "label": 1,
               },
               {
                   "sentence_1": "Children smiling and waving at camera",
                   "sentence_2": "The kids are frowning",
                   "label": 0,
               },
           ]
           train_dataset = Dataset.from_list(train_examples)
   
           # Define a loss function
           loss = BinaryCrossEntropyLoss(model)
   
           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(loss_fct)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6
   
           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               loss_fct=torch.nn.MSELoss(),
           )
        - .. code-block:: python
           :emphasize-lines: 1, 6, 7, 14
   
           from sentence_transformers.cross_encoder.losses import MSELoss

           ...

           # Prepare the loss function
           # See all valid losses in https://sbert.net/docs/cross_encoder/loss_overview.html
           loss = MSELoss(model)
   
           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(evaluator)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 9
   
           ...
   
           # Load an evaluator
           evaluator = CrossEncoderNanoBEIREvaluator()
   
           # Finetune with an evaluator
           model.fit(
               train_dataloader=train_dataloader,
               evaluator=evaluator,
           )
        - .. code-block:: python
           :emphasize-lines: 10
   
           # Load an evaluator
           evaluator = CrossEncoderNanoBEIREvaluator()
   
           # Finetune with an evaluator
           trainer = CrossEncoderTrainer(
               model=model,
               train_dataset=train_dataset,
               eval_dataset=eval_dataset,
               loss=loss,
               evaluator=evaluator,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(epochs)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6
   
           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               epochs=1,
           )
        - .. code-block:: python
           :emphasize-lines: 5
   
           ...

           # Prepare the Training Arguments
           args = CrossEncoderTrainingArguments(
               num_train_epochs=1,
           )
   
           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(activation_fct)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6
   
           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               activation_fct=torch.nn.Sigmoid(),
           )
        - .. code-block:: python
           :emphasize-lines: 4
   
           ...

           # Prepare the loss function
           loss = MSELoss(model, activation_fn=torch.nn.Sigmoid())
   
           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(scheduler)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6
   
           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               scheduler="WarmupLinear",
           )
        - .. code-block:: python
           :emphasize-lines: 6
   
           ...

           # Prepare the Training Arguments
           args = CrossEncoderTrainingArguments(
               # See https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.SchedulerType
               lr_scheduler_type="linear"
           )
   
           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(warmup_steps)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6
   
           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               warmup_steps=1000,
           )
        - .. code-block:: python
           :emphasize-lines: 5
   
           ...

           # Prepare the Training Arguments
           args = CrossEncoderTrainingArguments(
               warmup_steps=1000,
           )
   
           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(optimizer_class, optimizer_params)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6
   
           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               optimizer_class=torch.optim.AdamW,
               optimizer_params={"eps": 1e-7},
           )
        - .. code-block:: python
           :emphasize-lines: 6-7
   
           ...

           # Prepare the Training Arguments
           args = CrossEncoderTrainingArguments(
               # See https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
               optim="adamw_torch",
               optim_args={"eps": 1e-7},
           )
   
           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(weight_decay)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6
   
           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               weight_decay=0.02,
           )
        - .. code-block:: python
           :emphasize-lines: 5
   
           ...

           # Prepare the Training Arguments
           args = CrossEncoderTrainingArguments(
               weight_decay=0.02,
           )
   
           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(evaluation_steps)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6, 7
   
           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               evaluator=evaluator,
               evaluation_steps=1000,
           )
        - .. code-block:: python
           :emphasize-lines: 5, 6, 10, 15, 17
   
           ...

           # Prepare the Training Arguments
           args = CrossEncoderTrainingArguments(
               eval_strategy="steps",
               eval_steps=1000,
           )
   
           # Finetune the model
           # Note: You need an eval_dataset and/or evaluator to evaluate
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               eval_dataset=eval_dataset,
               loss=loss,
               evaluator=evaluator,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(output_path, save_best_model)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 7, 8
   
           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               evaluator=evaluator,
               output_path="my/path",
               save_best_model=True,
           )
        - .. code-block:: python
           :emphasize-lines: 5, 6, 19
   
           ...

           # Prepare the Training Arguments
           args = CrossEncoderTrainingArguments(
               load_best_model_at_end=True,
               metric_for_best_model="hotpotqa_ndcg@10", # E.g. `evaluator.primary_metric`
           )
   
           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

           # Save the best model at my output path
           model.save_pretrained("my/path")

.. collapse:: CrossEncoder.fit(max_grad_norm)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6
   
           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               max_grad_norm=1,
           )
        - .. code-block:: python
           :emphasize-lines: 5
   
           ...

           # Prepare the Training Arguments
           args = CrossEncoderTrainingArguments(
               max_grad_norm=1,
           )
   
           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(use_amp)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6
   
           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               use_amp=True,
           )
        - .. code-block:: python
           :emphasize-lines: 5, 6
   
           ...

           # Prepare the Training Arguments
           args = CrossEncoderTrainingArguments(
               fp16=True,
               bf16=False, # If your GPU supports it, you can also use bf16 instead
           )
   
           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(callback)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 3, 4, 9
   
           ...

           def printer_callback(score, epoch, steps):
               print(f"Score: {score:.4f} at epoch {epoch:d}, step {steps:d}")

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               callback=printer_callback,
           )
        - .. code-block:: python
           :emphasize-lines: 1, 5-10, 17
   
           from transformers import TrainerCallback

           ...

           class PrinterCallback(TrainerCallback):
               # Subclass any method from https://huggingface.co/docs/transformers/main_classes/callback#transformers.TrainerCallback
               def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                   print(f"Metrics: {metrics} at epoch {state.epoch:d}, step {state.global_step:d}")

           printer_callback = PrinterCallback()

           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               train_dataset=train_dataset,
               loss=loss,
               callbacks=[printer_callback],
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(show_progress_bar)

   .. list-table::
      :widths: 50 50
      :header-rows: 1
   
      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6
   
           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               show_progress_bar=True,
           )
        - .. code-block:: python
           :emphasize-lines: 5
   
           ...

           # Prepare the Training Arguments
           args = CrossEncoderTrainingArguments(
               disable_tqdm=False,
           )
   
           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. raw:: html
    
   <br>

.. note::

   The old :meth:`CrossEncoder.fit <sentence_transformers.cross_encoder.CrossEncoder.fit>` method still works, it was only softly deprecated. It now uses the new :class:`~sentence_transformers.cross_encoder.trainer.CrossEncoderTrainer` behind the scenes.
```


### Migration for CrossEncoder evaluators

```{eval-rst}
.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - v3.x
     - v4.x (recommended)
   * - ``CEBinaryAccuracyEvaluator``
     - Use :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderClassificationEvaluator`, an encompassed evaluator which uses the same inputs & outputs.
   * - ``CEBinaryClassificationEvaluator``
     - Use :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderClassificationEvaluator`, an encompassed evaluator which uses the same inputs & outputs.
   * - ``CECorrelationEvaluator``
     - Use :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderCorrelationEvaluator`, this evaluator was renamed.
   * - ``CEF1Evaluator``
     - Use :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderClassificationEvaluator`, an encompassed evaluator which uses the same inputs & outputs.
   * - ``CESoftmaxAccuracyEvaluator``
     - Use :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderClassificationEvaluator`, an encompassed evaluator which uses the same inputs & outputs.
   * - ``CERerankingEvaluator``
     - Renamed to :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderRerankingEvaluator`, this evaluator was renamed

.. note::

   The old evaluators still work, they will simply warn you to update to the new evaluators.
```