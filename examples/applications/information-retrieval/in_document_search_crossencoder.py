"""
This example show how in-document search can be used with a CrossEncoder.

The document is split into passage. Here, we use three consecutive sentences as a passage. You can use shorter passage, for example, individual sentences,
or longer passages, like full paragraphs.


The CrossEncoder takes the search query and scores every passage how relevant the passage is for the given score. The five passages with the highest score are  then returned.

As CrossEncoder, we use cross-encoder/ms-marco-TinyBERT-L-2, a BERT model with only 2 layers trained on the MS MARCO dataset. This is an extremely quick model able to score up to 9000 passages per second (on a V100 GPU). You can also use a larger model, which gives better results but is also slower.

Note: As we score the [query, passage]-pair for every new query, this search method
becomes at some point in-efficient if the document gets too large.

Usage: python in_document_search_crossencoder.py
"""

from sentence_transformers import CrossEncoder
from nltk import sent_tokenize
import time


#As document, we take the first two section from the Wikipedia article about Europe
document = """Europe is a continent located entirely in the Northern Hemisphere and mostly in the Eastern Hemisphere. It comprises the westernmost part of Eurasia and is bordered by the Arctic Ocean to the north, the Atlantic Ocean to the west, the Mediterranean Sea to the south, and Asia to the east. Europe is commonly considered to be separated from Asia by the watershed of the Ural Mountains, the Ural River, the Caspian Sea, the Greater Caucasus, the Black Sea, and the waterways of the Turkish Straits. Although some of this border is over land, Europe is generally accorded the status of a full continent because of its great physical size and the weight of history and tradition.

Europe covers about 10,180,000 square kilometres (3,930,000 sq mi), or 2% of the Earth's surface (6.8% of land area), making it the second smallest continent. Politically, Europe is divided into about fifty sovereign states, of which Russia is the largest and most populous, spanning 39% of the continent and comprising 15% of its population. Europe had a total population of about 741 million (about 11% of the world population) as of 2018. The European climate is largely affected by warm Atlantic currents that temper winters and summers on much of the continent, even at latitudes along which the climate in Asia and North America is severe. Further from the sea, seasonal differences are more noticeable than close to the coast.

European culture is the root of Western civilization, which traces its lineage back to ancient Greece and ancient Rome. The fall of the Western Roman Empire in 476 AD and the subsequent Migration Period marked the end of Europe's ancient history and the beginning of the Middle Ages. Renaissance humanism, exploration, art and science led to the modern era. Since the Age of Discovery, started by Portugal and Spain, Europe played a predominant role in global affairs. Between the 16th and 20th centuries, European powers colonized at various times the Americas, almost all of Africa and Oceania, and the majority of Asia.

The Age of Enlightenment, the subsequent French Revolution and the Napoleonic Wars shaped the continent culturally, politically and economically from the end of the 17th century until the first half of the 19th century. The Industrial Revolution, which began in Great Britain at the end of the 18th century, gave rise to radical economic, cultural and social change in Western Europe and eventually the wider world. Both world wars took place for the most part in Europe, contributing to a decline in Western European dominance in world affairs by the mid-20th century as the Soviet Union and the United States took prominence. During the Cold War, Europe was divided along the Iron Curtain between NATO in the West and the Warsaw Pact in the East, until the revolutions of 1989 and fall of the Berlin Wall.

In 1949, the Council of Europe was founded with the idea of unifying Europe to achieve common goals. Further European integration by some states led to the formation of the European Union (EU), a separate political entity that lies between a confederation and a federation. The EU originated in Western Europe but has been expanding eastward since the fall of the Soviet Union in 1991. The currency of most countries of the European Union, the euro, is the most commonly used among Europeans; and the EU's Schengen Area abolishes border and immigration controls between most of its member states. There exists a political movement favoring the evolution of the European Union into a single federation encompassing much of the continent.

In classical Greek mythology, Europa (Ancient Greek: Εὐρώπη, Eurṓpē) was a Phoenician princess. One view is that her name derives from the ancient Greek elements εὐρύς (eurús), "wide, broad" and ὤψ (ōps, gen. ὠπός, ōpós) "eye, face, countenance", hence their composite Eurṓpē would mean "wide-gazing" or "broad of aspect". Broad has been an epithet of Earth herself in the reconstructed Proto-Indo-European religion and the poetry devoted to it. An alternative view is that of R.S.P. Beekes who has argued in favor of a Pre-Indo-European origin for the name, explaining that a derivation from ancient Greek eurus would yield a different toponym than Europa. Beekes has located toponyms related to that of Europa in the territory of ancient Greece and localities like that of Europos in ancient Macedonia.

There have been attempts to connect Eurṓpē to a Semitic term for "west", this being either Akkadian erebu meaning "to go down, set" (said of the sun) or Phoenician 'ereb "evening, west", which is at the origin of Arabic Maghreb and Hebrew ma'arav. Michael A. Barry finds the mention of the word Ereb on an Assyrian stele with the meaning of "night, [the country of] sunset", in opposition to Asu "[the country of] sunrise", i.e. Asia. The same naming motive according to "cartographic convention" appears in Greek Ἀνατολή (Anatolḗ "[sun] rise", "east", hence Anatolia). Martin Litchfield West stated that "phonologically, the match between Europa's name and any form of the Semitic word is very poor", while Beekes considers a connection to Semitic languages improbable. Next to these hypotheses there is also a Proto-Indo-European root *h1regʷos, meaning "darkness", which also produced Greek Erebus.

Most major world languages use words derived from Eurṓpē or Europa to refer to the continent. Chinese, for example, uses the word Ōuzhōu (歐洲/欧洲), which is an abbreviation of the transliterated name Ōuluóbā zhōu (歐羅巴洲) (zhōu means "continent"); a similar Chinese-derived term Ōshū (欧州) is also sometimes used in Japanese such as in the Japanese name of the European Union, Ōshū Rengō (欧州連合), despite the katakana Yōroppa (ヨーロッパ) being more commonly used. In some Turkic languages, the originally Persian name Frangistan ("land of the Franks") is used casually in referring to much of Europe, besides official names such as Avrupa or Evropa."""

## We split this article into paragraphs and then every paragraph into sentences
paragraphs = []
for paragraph in document.replace("\r\n", "\n").split("\n\n"):
    if len(paragraph.strip()) > 0:
        paragraphs.append(sent_tokenize(paragraph.strip()))


#We combine up to 3 sentences into a passage. You can choose smaller or larger values for window_size
#Smaller value: Context from other sentences might get lost
#Lager values: More context from the paragraph remains, but results are longer
window_size = 3
passages = []
for paragraph in paragraphs:
    for start_idx in range(0, len(paragraph), window_size):
        end_idx = min(start_idx+window_size, len(paragraph))
        passages.append(" ".join(paragraph[start_idx:end_idx]))


print("Paragraphs: ", len(paragraphs))
print("Sentences: ", sum([len(p) for p in paragraphs]))
print("Passages: ", len(passages))


## Load our cross-encoder. Use fast tokenizer to speed up the tokenization
model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2')

## Some queries we want to search for in the document
queries = ["How large is Europe?",
           "Is Europe a continent?",
           "What is the currency in EU?",
           "Fall Roman Empire when",                    #We can also search for key word queries
           "Is Europa in the south part of the globe?"]   #Europe is miss-spelled & the matching sentences does not mention any of the content words

#Search in a loop for the individual queries
for query in queries:
    start_time = time.time()

    #Concatenate the query and all passages and predict the scores for the pairs [query, passage]
    model_inputs = [[query, passage] for passage in passages]
    scores = model.predict(model_inputs)

    #Sort the scores in decreasing order
    results = [{'input': inp, 'score': score} for inp, score in zip(model_inputs, scores)]
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    print("Query:", query)
    print("Search took {:.2f} seconds".format(time.time() - start_time))
    for hit in results[0:5]:
        print("Score: {:.2f}".format(hit['score']), "\t", hit['input'][1])


    print("==========")