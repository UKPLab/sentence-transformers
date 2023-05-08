# Training

This folder contains various examples to fine-tune `SentenceTransformers` for specific tasks.

For the beginning, I can recommend to have a look at the Semantic Textual Similarity ([STS](sts/)) or the Natural Language Inference ([NLI](nli/)) examples. 

For the documentation how to train your own models, see [Training Overview](http://www.sbert.net/docs/training/overview.html).

## Training Examples
- [avg_word_embeddings](avg_word_embeddings/) - This folder contains examples to train models based on classical word embeddings like GloVe. These models are extremely fast, but are a more inaccuracte than transformers based models.
- [distillation](distillation/) - Examples to make models smaller, faster and lighter.
- [multilingual](multilingual/) - Existent monolingual models can be extend to various languages ([paper](https://arxiv.org/abs/2004.09813)). This folder contains a step-by-step guide to extend existent models to new languages. 
- [nli](nli/) - Natural Language Inference (NLI) data can be quite helpful to pre-train and fine-tune models to create meaningful sentence embeddings.
- [quora_duplicate_questions](quora_duplicate_questions/) - Quora Duplicate Questions is large set corpus with duplicate questions from the Quora community. The folder contains examples how to train models for duplicate questions mining and for semantic search.
- [sts](sts/) - The most basic method to train models is using Semantic Textual Similarity (STS) data. Here, we have a sentence pair and a score indicating the semantic similarity.
- [other](other/) - Various tiny examples for show-casing one specific training case.

