'''
This file is used to generate query variations of mispelling
Besides the three original perturbations of Penha, I also referenced the code of DL_Typo, and the final complete version should be the same as DL_Typo
'''
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from textattack.augmentation import Augmenter
from textattack.transformations import *
from textattack.constraints.pre_transformation import (
    StopwordModification,
)
from textattack.constraints.pre_transformation import MinWordLength
from IPython import embed
from tqdm import tqdm

import pandas as pd
import logging 
from query.DL_Typo import FixWordSwapQWERTY

CONSTRAINTS = [StopwordModification(), MinWordLength(3)] # Compared to the original perturbations of Penha, I added the limit of MinWordLength(3) and it is consistent with DL_Typo

class MispellingActions():
    def __init__(self, queries, q_ids):
        self.queries = queries
        self.q_ids = q_ids
        self.augmenters = [
            Augmenter(transformation=WordSwapNeighboringCharacterSwap(), transformations_per_example=1, constraints=CONSTRAINTS),
            Augmenter(transformation=WordSwapRandomCharacterSubstitution(), transformations_per_example=1, constraints=CONSTRAINTS),
            Augmenter(transformation=WordSwapQWERTY(), transformations_per_example=1, constraints=CONSTRAINTS),
            # The following three are the perturbations added by DL_Typo
            Augmenter(transformation=WordSwapRandomCharacterDeletion(), transformations_per_example=1, constraints=CONSTRAINTS),
            Augmenter(transformation=WordSwapRandomCharacterInsertion(), transformations_per_example=1, constraints=CONSTRAINTS),
            Augmenter(transformation=FixWordSwapQWERTY(), transformations_per_example=1, constraints=CONSTRAINTS),
        ]

 

    def mispelling_chars(self, sample=None):
        logging.info("Adding mispelling errors using texttattack.")
        logging.info("Methods used: {}.".format(str([t.transformation.__class__.__name__ for t in self.augmenters])))
        i=0
        query_variations = []
        for query in tqdm(self.queries):
            for augmenter in self.augmenters:
                try:
                    augmented = augmenter.augment(query)
                except: #empty error for QWERTY.
                    augmented = [query]
                for q_variation in augmented:
                    query_variations.append([self.q_ids[i], query, q_variation, augmenter.transformation.__class__.__name__, "mispelling"])
            i+=1
            if sample and i > sample:
                break
        return query_variations