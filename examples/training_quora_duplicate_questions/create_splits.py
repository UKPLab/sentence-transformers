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
https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/quora-duplicate-questions.zip

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
random.seed(42)
random.shuffle(rows)

train_ids = set()
dev_ids = set()
test_ids = set()

counter = 0
for row in rows:
    if row['qid1'] in is_assigned or row['qid2'] in is_assigned:
        continue

    #Distribution about 85%/5%/10%
    target_set = train_ids
    if counter%10 == 0:
        target_set = dev_ids
    elif counter%10 == 1 or counter%10 == 2:
        target_set = test_ids

    #Get the sentence with all duplicates and add it to the respective sets
    target_set.add(row['qid1'])
    is_assigned.add(row['qid1'])

    target_set.add(row['qid2'])
    is_assigned.add(row['qid2'])

    for b in list(duplicates[row['qid1']])+list(duplicates[row['qid2']]):
        target_set.add(b)
        is_assigned.add(b)

    counter += 1

print("Train sentences:", len(train_ids))
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


print("Train duplicates", len(train_duplicates))
print("dev duplicates", len(dev_duplicates))
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


print("Write duplicate graph in pairwise format")
with open('quora-IR-dataset/graph/duplicates-graph-pairwise.tsv', 'w', encoding='utf8') as fOut:
    fOut.write("qid1\tqid2\n")
    for a, b in duplicates_list:
        fOut.write("{}\t{}\n".format(a, b))


print("Write duplicate graph in list format")
with open('quora-IR-dataset/graph/duplicates-graph-list.tsv', 'w', encoding='utf8') as fOut:
    fOut.write("qid1\tqid2\n")
    for a in sorted(duplicates.keys()):
        if len(duplicates[a]) > 0:
            fOut.write("{}\t{}\n".format(a, ",".join(sorted(duplicates[a]))))


def write_qids(name, ids_list):
    with open('quora-IR-dataset/graph/'+name+'-questions.tsv', 'w', encoding='utf8') as fOut:
        fOut.write("qid\n")
        fOut.write("\n".join(sorted(ids_list)))

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
for a, b in dev_duplicates:
    if a not in corpus_ids and b not in corpus_ids:
        if len(dev_queries) < num_dev_queries:
            dev_queries.add(a)
        else:
            corpus_ids.add(a)

        corpus_ids.add(b)
        for further_dups in duplicates[b]:
            if further_dups not in dev_queries:
                corpus_ids.add(further_dups)

#Create test queries
for a, b in test_duplicates:
    if a not in corpus_ids and b not in corpus_ids:
        if len(test_queries) < num_test_queries:
            test_queries.add(a)
        else:
            corpus_ids.add(a)

        corpus_ids.add(b)
        for further_dups in duplicates[b]:
            if further_dups not in test_queries:
                corpus_ids.add(further_dups)

#Write output for information-retrieval
print("Corpus size:", len(corpus_ids))
print("Dev queries:", len(dev_queries))
print("Test queries:", len(test_queries))

with open('quora-IR-dataset/information-retrieval/corpus.tsv', 'w', encoding='utf8') as fOut:
    fOut.write("qid\tquestion\n")
    for id in corpus_ids:
        fOut.write("{}\t{}\n".format(id, sentences[id]))

with open('quora-IR-dataset/information-retrieval/dev-queries.tsv', 'w', encoding='utf8') as fOut:
    fOut.write("qid\tquestion\tduplicate_qids\n")
    for id in dev_queries:
        fOut.write("{}\t{}\t{}\n".format(id, sentences[id], ",".join(duplicates[id])))

with open('quora-IR-dataset/information-retrieval/test-queries.tsv', 'w', encoding='utf8') as fOut:
    fOut.write("qid\tquestion\tduplicate_qids\n")
    for id in test_queries:
        fOut.write("{}\t{}\t{}\n".format(id, sentences[id], ",".join(duplicates[id])))






print("--DONE--")