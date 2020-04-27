# Extending Sentence Embeddings Models to New Languages
The issue with multilingual BERT (mBERT) as well as with XLM-RoBERTa is that those produce rather bad sentence representation out-of-the-box. Further, the vectors spaces between languages are not  aligned, i.e., the sentences with the same content in different languages would be mapped to different locations in the vector space.

In my publication [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813) I describe any easy approach to extend sentence embeddings to further languages.


## Available Pre-trained Models
- **distiluse-base-multilingual-cased**: Supported languages: Arabic, Chinese, Dutch, English, French, German,  Italian, Korean, Polish, Portuguese, Russian, Spanish, Turkish. Model is based on DistilBERT-multi-lingual.

## Usage
You can use the model in the following way:
```
embedder = SentenceTransformer('model-name')
embeddings = embedder.encode(['Hello World', 'Hallo Welt', 'Hola mundo'])
print(embeddings)
```


## Extend your own models
![Multilingual Knowledge Distillation](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/pretrained-models/multilingual-distillation.png)

The idea is based on a fixed (monolingual) **teacher model**, that produces sentence embeddings with our desired properties in one language. The **student model** is supposed to mimic the teacher model, i.e., the same English sentence should be mapped to the same vector by the teacher and by the student model. In order that the student model works for further languages, we train the student model on parallel (translated) sentences. The translation of each sentence should also be mapped to the same vector as the original sentence.

In the above figure, the student model should map *Hello World* and the German translation *Hallo Welt* to the vector of *teacher_model('Hello World')*. We achieve this by training the student model using mean squared error (MSE) loss.

In our experiments we initiliazed the student model with the multilingual XLM-RoBERTa model. 

**For an example**, how to extend an English model such that it works for English and German, see [training_sbert-en-de.py](https://github.com/UKPLab/sentence-transformers/tree/master/examples/training_multilingual/training_sbert-en-de.py)

I tested the approach with various languages with different alphabets, including Chinese and Arabic. The method allows to extend a model to multiple new languages in the same training process.


## Training Data
Your training data must be tab-seperated. In the first column, you have your source sentence, for example, an English sentence. In the following columns, you have the translations of this source sentence. If you have multiple translations per source sentence, you can put them in the same line or in different lines.
```
Source_sentence\tTarget_lang1\tTarget_lang2\tTarget_lang3
Source_sentence\tTarget_lang1\tTarget_lang2
```

An example file could look like this (EN DE ES):
```
Hello World Hallo Welt  Hola Mundo
Sentences are separated with a tab character.    Die Sätze sind per Tab getrennt.    Las oraciones se separan con un carácter de tabulación.
```

The order of the translations are not important.

You can load such a training file using the *ParallelSentencesDataset*
```
train_data = ParallelSentencesDataset(student_model=model, teacher_model=teacher_model)
train_data.load_data('path/to/tab/separated/train_file.tsv')

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MSELoss(model=model)
```

You load a file with the *load_data()* method. You can load multiple files by calling load_data multiple times. You can also pass a gzip compressed file to this method.

#Sources for Training Data
A great website for a vast number of parallel (translated) datasets is [OPUS](http://opus.nlpl.eu/). There, you find parallel datasets for more than 400 languages. 

## Performance
The performance was evaluated on the [Semantic Textual Similarity (STS) 2017 dataset](http://ixa2.si.ehu.es/stswiki/index.php/Main_Page). The task is to predict the semantic similarity (on a scale 0-5) of two given sentences. STS2017 has monolingual test data for English, Arabic, and Spanish, and cross-lingual test data for English-Arabic, -Spanish and -Turkish.

We extended the STS2017 and added cross-lingual test data for English-German, French-English, Italian-English, and Dutch-English ([STS2017-extended.zip](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/STS2017-extended.zip)). The performance is measured using Spearman correlation between the predicted similarity score and the gold score.

<table>
  <tr>
    <th>Model</th>
    <th>AR-AR</th>
    <th>AR-EN</th>
    <th>ES-ES</th>
    <th>ES-EN</th>
    <th>EN-EN</th>
    <th>TR-EN</th>
    <th>EN-DE</th>
    <th>FR-EN</th>
    <th>IT-EN</th>
    <th>NL-EN</th>
    <th>Average</th>
  </tr>
  <tr>
    <td>XLM-RoBERTa mean pooling </td>
    <td align="center">25.7</td>
    <td align="center">17.4</td>
    <td align="center">51.8</td>
    <td align="center">10.9</td>
    <td align="center">50.7</td>
    <td align="center">9.2</td>
    <td align="center">21.3</td>
    <td align="center">16.6</td>
    <td align="center">22.9</td>
    <td align="center">26.0</td>
    <td align="center">25.2</td>
  </tr>
  <tr>
    <td>mBERT mean pooling </td>
    <td align="center">50.9</td>
    <td align="center">16.7</td>
    <td align="center">56.7</td>
    <td align="center">21.5</td>
    <td align="center">54.4</td>
    <td align="center">16.0</td>
    <td align="center">33.9</td>
    <td align="center">33.0</td>
    <td align="center">34.0</td>
    <td align="center">35.6</td>
    <td align="center">35.3</td>
  </tr>
  <tr>
    <td>LASER</td>
    <td align="center">68.9</td>
    <td align="center">66.5</td>
    <td align="center">79.7</td>
    <td align="center">57.9</td>
    <td align="center">77.6</td>
    <td align="center">72.0</td>
    <td align="center">64.2</td>
    <td align="center">69.1</td>
    <td align="center">70.8</td>
    <td align="center">68.5</td>
    <td align="center">69.5</td>
  </tr> 
  <tr>
    <td colspan="9"><b>Sentence Transformer Models</b></td>
  </tr>
  <tr>
  <td>distiluse-base-multilingual-cased</td>
    <td align="center">75.9</td>
    <td align="center">77.6</td>
    <td align="center">85.3</td>
    <td align="center">78.7</td>
    <td align="center">85.4</td>
    <td align="center">75.5</td>
    <td align="center">80.3</td>
    <td align="center">80.2</td>
    <td align="center">80.5</td>
    <td align="center">81.7</td>
    <td align="center">80.1</td>
    </tr>
</table>

## Citation
If you use the code for multilingual models, feel free to cite our publication [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813):
``` 
@article{reimers-2020-multilingual-sentence-bert,
    title = "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation",
    author = "Reimers, Nils and Gurevych, Iryna",
    journal= "arXiv preprint arXiv:2004.09813",
    month = "04",
    year = "2020",
    url = "http://arxiv.org/abs/2004.09813",
}
```
