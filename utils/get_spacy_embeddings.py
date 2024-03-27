import json
import spacy
from gensim.models.keyedvectors import KeyedVectors
import sys
sys.path.append('.')

from utils.misc import get_data_dir

print('Downloading and converting spaCy German vectors ...')

nlp = spacy.load('de_core_news_lg')

word_list =[]
vectors = []
for key, vector in nlp.vocab.vectors.items():
    word_list.append(nlp.vocab.strings[key] )
    vectors.append(vector)

kv = KeyedVectors(nlp.vocab.vectors_length)

kv.add_vectors(word_list, vectors)

p = get_data_dir() / "tmp_models"

model_destination = p / 'spacy'

if not model_destination.exists():
    model_destination.mkdir(parents=True)

kv.save(str(model_destination / 'de_core_news_lg.model'))

print('Verify model can be loaded')
model = KeyedVectors.load(str(model_destination / 'de_core_news_lg'))

vocab_size = len(model.key_to_index)

print('Add model meta information')

model_meta = {
 "name": "de_core_news_lg",
 "training_corpus": "Various",
 "model_type": "neural",
 "learning_rate": 0.001,
 "epochs": 0,
 "word_ngrams": 1,
 "min_count": 0,
 "window_size": 9,
 "dimensions": 300,
 "parameter_string": "various_lr0.001_epochs0_mincount0_ws9_dims300",
 "model_path": str(model_destination / 'de_core_news_lg'),
 "vocab_size": vocab_size
}

with open(model_destination / "de_core_news_lg.json", "w") as f:
    json.dump(model_meta, f)

print('Done!')