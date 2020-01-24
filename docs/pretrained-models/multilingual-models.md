# Multilingual Pre-Trained Models
The issue with the multilingual BERT (mBERT) as well as with XLM-RoBERTa is that those produce rather bad sentence representation out-of-the-box. Especially, the vectors spaces are not well aligned. If an English and a Spanish sentence have the same meaning, these BERT produces different sentence embeddings. This is especially an issue for cross-lingual tasks.

Currently I train various multilingual models that can be used for producing sentence embeddings. The models have aligned vector spaces, i.e., independet of the language, sentences with similar meanings should be mapped to the same point in vector space.


## Available models:
- **distiluse-base-multilingual-cased**: Supported languages: Arabic, Chinese, Dutch, English, French, German,  Italian, Korean, Polish, Portuguese, Russian, Spanish, Turkish. Model is based on DistilBERT-multi-lingual.

Further models and details will be added soon.

## Usage
You can use the model in the following way:
```
embedder = SentenceTransformer('model-name')
embeddings = embedder.encode(['Hello World', 'Hallo Welt', 'Hola mundo'])
print(embeddings)
```


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


