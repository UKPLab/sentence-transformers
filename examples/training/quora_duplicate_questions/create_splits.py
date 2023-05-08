"""
The Quora Duplicate Questions dataset contains questions pairs from Quora (www.quora.com)
along with a label whether the two questions are a duplicate, i.e., have an identical itention.

Example of a duplicate pair:
How do I enhance my English?  AND  How can I become good at English?

Example of a non-duplicate pair:
How are roads named?   AND    How are airport runways named?

More details and the original Quora dataset can be found here:
https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs
Dataset: http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv

You do not need to run this script. You can download all files from here:
https://sbert.net/datasets/quora-duplicate-questions.zip

This script does the following:
1) After reading the quora_duplicate_questions.tsv, as provided by Quora, we add a transitive closure: If question (A, B) are duplicates and (B, C) are duplicates, than (A, C) must also be a duplicate. We add these missing links.

2) Next, we split sentences into train, dev, and test with a ratio of about 85% / 5% / 10%. In contrast to must other Quora data splits, like the split provided by GLUE, we ensure that the three sets are overlap free, i.e., no sentences in dev / test will appear in the train dataset. In order to achieve three distinct datasets, we pick a sentence and then assign all duplicate sentences to this dataset to that repective set

3) After distributing sentences to the three dataset split, we create files to facilitate 3 different tasks:
    3.1) Classification - Given two sentences, are these a duplicate? This is identical to the orginial Quora task and the task in GLUE, but with the big difference that sentences in dev / test have not been seen in train.
    3.2) Duplicate Question Mining - Given a large set of questions, identify all duplicates. The dev set consists of about 50k questions, the test set of about 100k sentences.
    3.3) Information Retrieval - Given a question as query, find in a large corpus (~350k questions) the duplicates of the query question.


The output consists of the following files:

quora_duplicate_questions.tsv - Original file provided by Quora (https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)

classification/
    train/dev/test_pairs.tsv - Distinct sets of question pairs with label for duplicate / non-duplicate. These splits can be used for sentence pair classification tasks

duplicate-mining/ - Given a large set of questions, find all duplicates.
    _corpus.tsv - Large set of sentences
    _duplicates.tsv - All duplicate questions in the respective corpus.tsv

information-retrieval/  - Given a large corpus of questions, find the duplicates for a given query
    corpus.tsv - This file will be used for train/dev/test. It contains all questions in the corpus
    dev/test-queries.tsv - Queries and the respective duplicate questions (QIDs) in the corpus

"""
import csv
from collections import defaultdict
import random
import os
from sentence_transformers import util


random.seed(42)

#Get raw file
source_file = "quora-IR-dataset/quora_duplicate_questions.tsv"
os.makedirs('quora-IR-dataset', exist_ok=True)
os.makedirs('quora-IR-dataset/graph', exist_ok=True)
os.makedirs('quora-IR-dataset/information-retrieval', exist_ok=True)
os.makedirs('quora-IR-dataset/classification', exist_ok=True)
os.makedirs('quora-IR-dataset/duplicate-mining', exist_ok=True)

if not os.path.exists(source_file):
    print("Download file to", source_file)
    util.http_get('http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv', source_file)

#Read pairwise file
sentences = {}
duplicates = defaultdict(lambda: defaultdict(bool))
rows = []
with open(source_file, encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        id1 = row['qid1']
        id2 = row['qid2']
        question1 = row['question1'].replace("\r", "").replace("\n", " ").replace("\t", " ")
        question2 = row['question2'].replace("\r", "").replace("\n", " ").replace("\t", " ")
        is_duplicate = row['is_duplicate']

        if question1 == "" or question2 == "":
            continue

        sentences[id1] = question1
        sentences[id2] = question2

        rows.append({'qid1': id1, 'qid2': id2, 'question1': question1, 'question2': question2, 'is_duplicate': is_duplicate})

        if is_duplicate == '1':
            duplicates[id1][id2] = True
            duplicates[id2][id1] = True


# Search for (near) exact duplicates
# The original Quora duplicate questions dataset is an incomplete annotation,
# i.e., there are several duplicate question pairs which are not marked as duplicates.
# These missing annotation can make it difficult to compare approaches.
# Here we use a simple approach that searches for near identical questions, that only differ in maybe a stopword
# We mark these found question pairs also as duplicate to increase the annotation coverage
stopwords = set(['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't", "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'which', 'while', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'])

num_new_duplicates = 0
sentences_norm = {}

for id, sent in sentences.items():
    sent_norm = sent.lower()

    #Replace some common paraphrases
    sent_norm = sent_norm.replace("how do you", "how do i").replace("how do we", "how do i")
    sent_norm = sent_norm.replace("how can we", "how can i").replace("how can you", "how can i").replace("how can i", "how do i")
    sent_norm = sent_norm.replace("really true", "true")
    sent_norm = sent_norm.replace("what are the importance", "what is the importance")
    sent_norm = sent_norm.replace("what was", "what is")
    sent_norm = sent_norm.replace("so many", "many")
    sent_norm = sent_norm.replace("would it take", "will it take")

    #Remove any punctuation characters
    for c in [",", "!", ".", "?", "'", '"', ":", ";", "[", "]", "{", "}", "<", ">"]:
        sent_norm = sent_norm.replace(c, " ")

    #Remove stop words
    tokens = sent_norm.split()
    tokens = [token for token in tokens if token not in stopwords]
    sent_norm = "".join(tokens)


    if sent_norm in sentences_norm:
        if not duplicates[id][sentences_norm[sent_norm]]:
            num_new_duplicates += 1

        duplicates[id][sentences_norm[sent_norm]] = True
        duplicates[sentences_norm[sent_norm]][id] = True
    else:
        sentences_norm[sent_norm] = id


print("(Nearly) exact duplicates found:", num_new_duplicates)


#Add transitive closure (if a,b and b,c duplicates => a,c are duplicates)
new_entries = True
while new_entries:
    print("Add transitive closure")
    new_entries = False
    for a in sentences:
        for b in list(duplicates[a]):
            for c in list(duplicates[b]):
                if a != c and not duplicates[a][c]:
                    new_entries = True
                    duplicates[a][c] = True
                    duplicates[c][a] = True


#Distribute rows to train/dev/test split
#Ensure that sets contain distinct sentences
is_assigned = set()
random.shuffle(rows)

train_ids = set()
dev_ids = set()
test_ids = set()

counter = 0
for row in rows:
    if row['qid1'] in is_assigned and row['qid2'] in is_assigned:
        continue
    elif row['qid1'] in is_assigned or row['qid2'] in is_assigned:

        if row['qid2'] in is_assigned: #Ensure that qid1 is assigned and qid2 not yet
            row['qid1'], row['qid2'] = row['qid2'], row['qid1']

        #Move qid2 to the same split as qid1
        target_set = train_ids
        if row['qid1'] in dev_ids:
            target_set = dev_ids
        elif row['qid1'] in test_ids:
            target_set = test_ids

    else:
        #Distribution about 85%/5%/10%
        target_set = train_ids
        if counter%10 == 0:
            target_set = dev_ids
        elif counter%10 == 1 or counter%10 == 2:
            target_set = test_ids
        counter += 1

    #Get the sentence with all duplicates and add it to the respective sets
    target_set.add(row['qid1'])
    is_assigned.add(row['qid1'])

    target_set.add(row['qid2'])
    is_assigned.add(row['qid2'])

    for b in list(duplicates[row['qid1']])+list(duplicates[row['qid2']]):
        target_set.add(b)
        is_assigned.add(b)


#Assert all sets are mutually exclusive
assert len(train_ids.intersection(dev_ids)) == 0
assert len(train_ids.intersection(test_ids)) == 0
assert len(test_ids.intersection(dev_ids)) == 0


print("\nTrain sentences:", len(train_ids))
print("Dev sentences:", len(dev_ids))
print("Test sentences:", len(test_ids))

#Extract the ids for duplicate questions for train/dev/test
def get_duplicate_set(ids_set):
    dups_set = set()
    for a in ids_set:
        for b in duplicates[a]:
            ids = sorted([a,b])
            dups_set.add(tuple(ids))
    return dups_set

train_duplicates = get_duplicate_set(train_ids)
dev_duplicates = get_duplicate_set(dev_ids)
test_duplicates = get_duplicate_set(test_ids)


print("\nTrain duplicates", len(train_duplicates))
print("Dev duplicates", len(dev_duplicates))
print("Test duplicates", len(test_duplicates))

############### Write general files about the duplate questions graph ############
with open('quora-IR-dataset/graph/sentences.tsv', 'w', encoding='utf8') as fOut:
    fOut.write("qid\tquestion\n")
    for id, question in sentences.items():
        fOut.write("{}\t{}\n".format(id, question))

duplicates_list = set()
for a in duplicates:
    for b in duplicates[a]:
        duplicates_list.add(tuple(sorted([int(a), int(b)])))


duplicates_list = list(duplicates_list)
duplicates_list = sorted(duplicates_list, key=lambda x: x[0]*1000000+x[1])


print("\nWrite duplicate graph in pairwise format")
with open('quora-IR-dataset/graph/duplicates-graph-pairwise.tsv', 'w', encoding='utf8') as fOut:
    fOut.write("qid1\tqid2\n")
    for a, b in duplicates_list:
        fOut.write("{}\t{}\n".format(a, b))


print("Write duplicate graph in list format")
with open('quora-IR-dataset/graph/duplicates-graph-list.tsv', 'w', encoding='utf8') as fOut:
    fOut.write("qid1\tqid2\n")
    for a in sorted(duplicates.keys(), key=lambda x: int(x)):
        if len(duplicates[a]) > 0:
            fOut.write("{}\t{}\n".format(a, ",".join(sorted(duplicates[a]))))

print("Write duplicate graph in connected subgraph format")
with open('quora-IR-dataset/graph/duplicates-graph-connected-nodes.tsv', 'w', encoding='utf8') as fOut:
    written_qids = set()
    fOut.write("qids\n")
    for a in sorted(duplicates.keys(), key=lambda x: int(x)):
        if a not in written_qids:
            ids = set()
            ids.add(a)

            for b in duplicates[a]:
                ids.add(b)

            fOut.write("{}\n".format(",".join(sorted(ids, key=lambda x: int(x)))))
            for id in ids:
                written_qids.add(id)

def write_qids(name, ids_list):
    with open('quora-IR-dataset/graph/'+name+'-questions.tsv', 'w', encoding='utf8') as fOut:
        fOut.write("qid\n")
        fOut.write("\n".join(sorted(ids_list, key=lambda x: int(x))))

write_qids('train', train_ids)
write_qids('dev', dev_ids)
write_qids('test', test_ids)


####### Output for duplicate mining #######
def write_mining_files(name, ids, dups):
    with open('quora-IR-dataset/duplicate-mining/'+name+'_corpus.tsv', 'w', encoding='utf8') as fOut:
        fOut.write("qid\tquestion\n")
        for id in ids:
            fOut.write("{}\t{}\n".format(id, sentences[id]))

    with open('quora-IR-dataset/duplicate-mining/'+name+'_duplicates.tsv', 'w', encoding='utf8') as fOut:
        fOut.write("qid1\tqid2\n")
        for a, b in dups:
            fOut.write("{}\t{}\n".format(a, b))


write_mining_files('train', train_ids, train_duplicates)
write_mining_files('dev', dev_ids, dev_duplicates)
write_mining_files('test', test_ids, test_duplicates)


###### Classification dataset #####
with open('quora-IR-dataset/classification/train_pairs.tsv', 'w', encoding='utf8') as fOutTrain, open('quora-IR-dataset/classification/dev_pairs.tsv', 'w', encoding='utf8') as fOutDev, open('quora-IR-dataset/classification/test_pairs.tsv', 'w', encoding='utf8') as fOutTest:
    fOutTrain.write("\t".join(['qid1', 'qid2', 'question1', 'question2', 'is_duplicate'])+"\n")
    fOutDev.write("\t".join(['qid1', 'qid2', 'question1', 'question2', 'is_duplicate']) + "\n")
    fOutTest.write("\t".join(['qid1', 'qid2', 'question1', 'question2', 'is_duplicate']) + "\n")

    for row in rows:
        id1 = row['qid1']
        id2 = row['qid2']

        target = None
        if id1 in train_ids and id2 in train_ids:
            target = fOutTrain
        elif id1 in dev_ids and id2 in dev_ids:
            target = fOutDev
        elif id1 in test_ids and id2 in test_ids:
            target = fOutTest

        if target is not None:
            target.write("\t".join([row['qid1'], row['qid2'], sentences[id1], sentences[id2], row['is_duplicate']]))
            target.write("\n")


####### Write files for Information Retrieval #####
num_dev_queries = 5000
num_test_queries = 10000

corpus_ids = train_ids.copy()
dev_queries = set()
test_queries = set()

#Create dev queries
rnd_dev_ids = sorted(list(dev_ids))
random.shuffle(rnd_dev_ids)

for a in rnd_dev_ids:
    if a not in corpus_ids:
        if len(dev_queries) < num_dev_queries and len(duplicates[a]) > 0:
            dev_queries.add(a)
        else:
            corpus_ids.add(a)

        for b in duplicates[a]:
            if b not in dev_queries:
                corpus_ids.add(b)

#Create test queries
rnd_test_ids = sorted(list(test_ids))
random.shuffle(rnd_test_ids)

for a in rnd_test_ids:
    if a not in corpus_ids:
        if len(test_queries) < num_test_queries and len(duplicates[a]) > 0:
            test_queries.add(a)
        else:
            corpus_ids.add(a)

        for b in duplicates[a]:
            if b not in test_queries:
                corpus_ids.add(b)

#Write output for information-retrieval
print("\nInformation Retrival Setup")
print("Corpus size:", len(corpus_ids))
print("Dev queries:", len(dev_queries))
print("Test queries:", len(test_queries))

with open('quora-IR-dataset/information-retrieval/corpus.tsv', 'w', encoding='utf8') as fOut:
    fOut.write("qid\tquestion\n")
    for id in sorted(corpus_ids, key=lambda id: int(id)):
        fOut.write("{}\t{}\n".format(id, sentences[id]))

with open('quora-IR-dataset/information-retrieval/dev-queries.tsv', 'w', encoding='utf8') as fOut:
    fOut.write("qid\tquestion\tduplicate_qids\n")
    for id in sorted(dev_queries, key=lambda id: int(id)):
        fOut.write("{}\t{}\t{}\n".format(id, sentences[id], ",".join(duplicates[id])))

with open('quora-IR-dataset/information-retrieval/test-queries.tsv', 'w', encoding='utf8') as fOut:
    fOut.write("qid\tquestion\tduplicate_qids\n")
    for id in sorted(test_queries, key=lambda id: int(id)):
        fOut.write("{}\t{}\t{}\n".format(id, sentences[id], ",".join(duplicates[id])))


print("--DONE--")





