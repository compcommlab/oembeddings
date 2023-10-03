import json
import gzip
import zipfile
from pathlib import Path

from tqdm import tqdm
import requests
import fasttext

import sys
sys.path.append('.')

from utils.misc import get_data_dir

p = get_data_dir() / "tmp_models"

def downloader(url: str, destination: Path, **kwargs):
    outputfile = destination / url.split('/')[-1]
    if outputfile.exists():
        print('Data already downloaded, not downloading again.')
    else:
        response = requests.get(url, stream=True, **kwargs)
        total_size = int(response.headers.get('content-length', 0))
        with open(outputfile, 'wb') as f:
            with tqdm(total=total_size, unit="byte") as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    b = f.write(chunk)
                    pbar.update(b)


print('Downloading fastText Common-Crawl German vectors ...')

model_destination = p / 'cc_de_300'

if not model_destination.exists():
    model_destination.mkdir()

downloader("https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.bin.gz",
           model_destination)

downloader("https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.vec.gz",
           model_destination)

print('decompressing ...')

for gz in model_destination.glob('*.gz'):
    decompressed = model_destination / gz.name.replace('.gz', '')
    with open(decompressed, "wb") as f_out:
        with gzip.open(gz, "r") as f_in:
            f_out.write(f_in.read())

print('Verify model can be loaded')
model = fasttext.load_model(str(model_destination / "cc.de.300.bin"))

vocab_size = len(model.words)

print('Add model meta information')

model_meta = {
 "name": "cc_de_300",
 "training_corpus": "Common Crawl",
 "model_type": "cbow",
 "learning_rate": 0.1,
 "epochs": 1,
 "word_ngrams": 1,
 "min_count": 5,
 "window_size": 5,
 "dimensions": 300,
 "parameter_string": "commoncrawl_lr0.1_epochs5_mincount1_ws5_dims300",
 "model_path": str(model_destination / 'cc.de.300'),
 "vocab_size": vocab_size
}

with open(model_destination / "cc.de.300.json", "w") as f:
    json.dump(model_meta, f)

########################
# Wikipedia based models
########################

print('Downloading fastText Wikipedia German vectors ...')

model_destination = p / 'wiki_de_300'

if not model_destination.exists():
    model_destination.mkdir()


downloader("https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.de.zip",
           model_destination)

downloader("https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.de.vec",
           model_destination)

print('decompressing ...')

z = zipfile.ZipFile(model_destination / "wiki.de.zip", mode='r')
z.extractall(path=model_destination)

print('Verify model can be loaded')
model = fasttext.load_model(str(model_destination / "wiki.de.bin"))

vocab_size = len(model.words)

print('Add model meta information')

model_meta = {
 "name": "wiki_de_300",
 "training_corpus": "Wikipedia",
 "model_type": "skipgram",
 "learning_rate": 0.1,
 "epochs": 1,
 "word_ngrams": 1,
 "min_count": 5,
 "window_size": 5,
 "dimensions": 300,
 "parameter_string": "wikipedia_lr0.1_epochs5_mincount1_ws5_dims300",
 "model_path": str(model_destination / 'wiki.de'),
 "vocab_size": vocab_size
}

with open(model_destination / "wiki.de.json", "w") as f:
    json.dump(model_meta, f)


print('Done!')