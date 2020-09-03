import urllib.request
import zipfile
import os
folder_path = os.path.dirname(os.path.realpath(__file__))
print('Beginning download of datasets')

datasets = ['AllNLI.zip', 'stsbenchmark.zip', 'wikipedia-sections-triplets.zip', 'STS2017.en-de.txt.gz', 'TED2013-en-de.txt.gz', 'xnli-en-de.txt.gz']
server = "https://sbert.net/datasets/"

for dataset in datasets:
    print("Download", dataset)
    url = server+dataset
    dataset_path = os.path.join(folder_path, dataset)
    urllib.request.urlretrieve(url, dataset_path)

    if dataset.endswith('.zip'):
        print("Extract", dataset)
        with zipfile.ZipFile(dataset_path, "r") as zip_ref:
            zip_ref.extractall(folder_path)
        os.remove(dataset_path)


print("All datasets downloaded and extracted")
