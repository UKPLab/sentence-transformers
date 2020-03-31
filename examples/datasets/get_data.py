import argparse
import urllib.request
import zipfile
import os

datasets = ['AllNLI.zip', 'stsbenchmark.zip', 'wikipedia-sections-triplets.zip']
server = "https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default=os.path.dirname(os.path.realpath(__file__)))
    args = parser.parse_args()
    print('Beginning download of datasets')

    for dataset in datasets:
        print("Downloading", dataset)
        url = server + dataset
        dataset_path = os.path.join(args.output_path, dataset)
        urllib.request.urlretrieve(url, dataset_path)

        print("Extracting", dataset)
        with zipfile.ZipFile(dataset_path, "r") as zip_ref:
            zip_ref.extractall(args.output_path)
        os.remove(dataset_path)
    print("All datasets downloaded and extracted")
