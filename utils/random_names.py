""" 
    Utility for generating random names for models 
    It takes the most common names in Austria and attaches 
    an adjective to it.
    Don't take this module too seriously :) 
"""

import pandas as pd
from pathlib import Path
import os
import random
from typing import Literal

p = Path(os.path.split(__file__)[0])

# Names are the 2000 most common first names in Austria
# Taken from https://www.statistik.at/atlas/vornamen/
NAMES = pd.read_csv(p / 'names.csv')

# Most positive adjectives
# Taken from the Berlin Affective Word List Reloaded (Võ et al 2009)
# DOI: 10.3758/BRM.41.2.534
# Link to Dataset: https://osf.io/hx6r8/
# Plus some other adjectives
ADJECTIVES = [
"aktiv",
"angeregt",
"beliebt",
"brillant",
"freudig",
"genial",
"grandios",
"kreativ",
"lebendig",
"makellos",
"munter",
"mutig",
"perfekt",
"reich",
"reizvoll",
"sinnlich",
"spannend",
"spontan",
"spritzig",
"stark",
"stilvoll",
"tapfer",
"toll",
"topfit",
"vital",
"wachsam", 
"grantig", # Start of additional adjectives
"urig",
"pfiffig",
"flott",
"gefinkelt",
"fesch",
"griawig",
"wunderbar",
"wundervoll",
"bezaubernd",
"entzückend",
"gescheit"
]

def generate_random_name(gender: Literal['female', 'male'] = None) -> str:
    if gender:
        name = NAMES[NAMES.gender == gender].sample(1).to_dict(orient='records')[0]
    else:
        name = NAMES.sample(1).to_dict(orient='records')[0]
    adjective = random.sample(ADJECTIVES, 1)[0]
    if name['gender'] == 'male':
        adjective += 'er'
    else:
        adjective += 'e'

    return adjective + '_' + name['name'].lower()


if __name__ == '__main__':
    print('testing random name generation')

    for i in range(10):
        print(generate_random_name())

    print()
    print('female only')
    for i in range(10):
        print(generate_random_name(gender='female'))