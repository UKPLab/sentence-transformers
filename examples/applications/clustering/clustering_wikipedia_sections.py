"""
This examples clusters different sentences that come from the same wikipedia article.

It uses the 'wikipedia-sections' model, a model that was trained to differentiate if two sentences from the
same article come from the same section or from different sections in that article.
"""
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering



embedder = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens')

#Sentences and sections are from Wikipeda.
#Source: https://en.wikipedia.org/wiki/Bushnell,_Illinois
corpus = [
("Bushnell is located at 40°33′6″N 90°30′29″W (40.551667, -90.507921).", "Geography"),
("According to the 2010 census, Bushnell has a total area of 2.138 square miles (5.54 km2), of which 2.13 square miles (5.52 km2) (or 99.63%) is land and 0.008 square miles (0.02 km2) (or 0.37%) is water.", "Geography"),

("The town was founded in 1854 when the Northern Cross Railroad built a line through the area.", "History"),
("Nehemiah Bushnell was the President of the Railroad, and townspeople honored him by naming their community after him. ", "History"),
("Bushnell was also served by the Toledo, Peoria and Western Railway, now the Keokuk Junction Railway.", "History"),

("As of the census[6] of 2000, there were 3,221 people, 1,323 households, and 889 families residing in the city. ", "Demographics"),
("The population density was 1,573.9 people per square mile (606.7/km²).", "Demographics"),
("There were 1,446 housing units at an average density of 706.6 per square mile (272.3/km²).", "Demographics"),

("From 1991 to 2012, Bushnell was home to one of the largest Christian Music and Arts festivals in the world, known as the Cornerstone Festival.", "Music"),
("Each year around the 4th of July, 25,000 people from all over the world would descend on the small farm town to watch over 300 bands, authors and artists perform at the Cornerstone Farm Campgrounds.", "Music"),
("The festival was generally well received by locals, and businesses in the area would typically put up signs welcoming festival-goers to their town.", "Music"),
("As a result of the location of the music festival, numerous live albums and videos have been recorded or filmed in Bushnell, including the annual Cornerstone Festival DVD. ", "Music"),
("Cornerstone held its final festival in 2012 and no longer operates.", "Music"),

("Beginning in 1908, the Truman Pioneer Stud Farm in Bushnell was home to one of the largest horse shows in the Midwest.", "Horse show"),
("The show was well known for imported European horses.", "Horse show"),
("The Bushnell Horse Show features some of the best Belgian and Percheron hitches in the country. Teams have come from many different states and Canada to compete.", "Horse show"),
]

sentences = [row[0] for row in corpus]

corpus_embeddings = embedder.encode(sentences)
num_clusters = len(set([row[1] for row in corpus]))

#Sklearn clustering
km = AgglomerativeClustering(n_clusters=num_clusters)
km.fit(corpus_embeddings)

cluster_assignment = km.labels_


clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i+1)
    for row in cluster:
        print("(Gold label: {}) - {}".format(row[1], row[0]))
    print("")

