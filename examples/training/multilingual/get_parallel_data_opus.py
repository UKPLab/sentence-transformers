"""
OPUS (http://opus.nlpl.eu/) is a great collection of different parallel datasets for more than 400 languages.
On the website, you can download parallel datasets for many languages in different formats. I found that
the format "Bottom-left triangle: download plain text files (MOSES/GIZA++)"  requires minimal
overhead for post-processing to get it into a suitable format for this library.

You can use the OPUS dataset to create multilingual sentence embeddings. This script contains code to download
OPUS datasets for the desired languages and to create training files in the right format.

1) First, you need to install OpusTools (https://github.com/Helsinki-NLP/OpusTools/tree/master/opustools_pkg):
pip install opustools

2) Once you have OpusTools installed, you can download data in the right format via:
mkdir parallel-sentences
opus_read -d [CORPUS] -s [SRC_LANG] -t [TRG_LANG] --write parallel-sentences/[FILENAME].tsv.gz   -wm moses -dl opus -p raw

For example:
mkdir parallel-sentences
opus_read -d JW300 -s en -t de --write parallel-sentences/JW300-en-de.tsv.gz -wm moses -dl opus -p raw

This downloads the JW300 Corpus (http://opus.nlpl.eu/JW300.php) for English (en) and German (de) and write the output to
parallel-sentences/JW300-en-de.tsv.gz


####################

This python code automates the download and creation of the parallel sentences files.


"""
from opustools import OpusRead
import os


corpora = ['JW300']  #Corpora you want to use
source_languages = ['en']            #Source language, our teacher model is able to understand
target_languages = ['de', 'es', 'it', 'fr', 'ar', 'tr']    #Target languages, out student model should learn

output_folder = 'parallel-sentences'
opus_download_folder = './opus'

#Iterator over all corpora / source languages / target languages combinations and download files
os.makedirs(output_folder, exist_ok=True)

for corpus in corpora:
    for src_lang in source_languages:
        for trg_lang in target_languages:
            output_filename = os.path.join(output_folder, "{}-{}-{}.tsv.gz".format(corpus, src_lang, trg_lang))
            if not os.path.exists(output_filename):
                print("Create:", output_filename)
                try:
                    read = OpusRead(directory=corpus, source=src_lang, target=trg_lang, write=[output_filename], download_dir=opus_download_folder, preprocess='raw', write_mode='moses', suppress_prompts=True)
                    read.printPairs()
                except:
                    print("An error occured during the creation of", output_filename)