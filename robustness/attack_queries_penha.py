'''
The purpose of this python file is to complete data augmentation in advance, because I found that it is very time-consuming sometimes.

# 2025-11-14 Updated code, used to generate query variations
# 2025-11-15 Updated code, used to generate Penha query variations, there are 5 types of variations

'''
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from attack.Penha_query_variation_generation.transformations_synonym import SynonymActions
from attack.Penha_query_variation_generation.transformations_paraphrase import ParaphraseActions
from attack.Penha_query_variation_generation.transformations_naturality import NaturalityActions
from attack.Penha_query_variation_generation.transformations_mispelling import MispellingActions
from attack.Penha_query_variation_generation.transformations_ordering import OrderingActions

from tqdm import tqdm

import time
import torch
import json
import random

from transformers.trainer_utils import set_seed
import wandb
 
import argparse

from utils.load_data import load_beir_data
from utils.logging import LoggingHandler

import logging
import os
import json
import torch
import glob
import pandas as pd
from attack.query.typo_text_attack import MixTextAttack,CharacterTextAttack, WordTextAttack
from attack.query.DL_Typo import dl_typo_beir
from typing import Any, Dict, List, Sequence

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#### /print debug information to stdout

DEFAULT_UQV_MODEL_PATH = "./"


def config_parse():
    # Parse command line arguments for evaluation
    parser = argparse.ArgumentParser(description="Evaluate BEIR retrieval results with different LLM-based IR models.")
    parser.add_argument("--dataset", type=str, default="fiqa", help="Dataset to evaluate on (e.g., 'nq', 'msmarco' arguana nfcorpus scifact scidocs fiqa trec-covid)" )
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use for evaluation (default: 'test').")
    parser.add_argument('--seed', type=int, default=1999, help='Seed for evaluation.')
 
    parser.add_argument("--attack_word_change_rate", type=float, default=0.2, help="Attack word change rate for the attack.")
    parser.add_argument("--attack_method", type=str, default="naturality", help="Attack method for the attack. mispelling, ordering, synonym, paraphrase, naturality  ")
    parser.add_argument(
        "--uqv_model_path",
        type=str,
        default=DEFAULT_UQV_MODEL_PATH,
        help="Path to store/load custom UQV paraphrase checkpoints (optional).",
    )

    args = parser.parse_args()
    return args

DEFAULT_ADV_SCORES_ROOT =  os.path.join(os.path.abspath(os.path.dirname(__file__)), 'output_attack' )


def _choose_single_variation_per_query(
    variations: Sequence[Sequence],
    original_queries: Dict[str, str],
) -> Dict[str, str]:
    """
    For every query ID, randomly select a single rewritten query.
    Falls back to the original text if no variation is available.
    """
    variations_per_qid: Dict[str, List[str]] = {}
    for entry in variations:
        if len(entry) < 3:
            continue
        q_id = str(entry[0])
        rewritten_text = entry[2]
        variations_per_qid.setdefault(q_id, []).append(rewritten_text)

    selected: Dict[str, str] = {}
    for q_id, original_text in original_queries.items():
        candidates = variations_per_qid.get(str(q_id))
        if candidates:
            selected[str(q_id)] = random.choice(candidates)
        else:
            selected[str(q_id)] = original_text
            print(f"No variation found for query {q_id}")
    return selected


def save_attacked_queries(
    variations,
    original_queries: Dict[str, str],
    output_root,
    config,
    *,
    subdir='attacked_text/query',
    include_wordrate=False,
):
    """
    Persist attacked queries (one sampled variation per query) as JSON, and dump
    all generated variations to a CSV for inspection/debugging.
    """
    attacked_queries = _choose_single_variation_per_query(variations, original_queries)
    filename = f"{config.dataset}_{config.split}_attacked_queries_seed_{config.seed}_attack_method_{config.attack_method}"
    if include_wordrate:
        filename = f"{filename}_wordrate_{config.attack_word_change_rate}"

    save_dir = os.path.join(output_root, subdir, config.dataset)
    os.makedirs(save_dir, exist_ok=True)

    attacked_queries_json_path = os.path.join(save_dir, f"{filename}.json")
    with open(attacked_queries_json_path, 'w', encoding='utf-8') as f:
        json.dump(attacked_queries, f, indent=4, ensure_ascii=False)
    print(f"Saved attacked queries to {attacked_queries_json_path} in json format")

    csv_rows = []
    for entry in variations:
        if len(entry) < 3:
            continue
        q_id = str(entry[0])
        raw_query = entry[1] if len(entry) > 1 else original_queries.get(q_id, "")
        attacked_query = entry[2]
        attack_method = entry[3] if len(entry) > 3 else config.attack_method
        csv_rows.append(
            {
                "queryid": q_id,
                "raw_query": raw_query,
                "attacked_query": attacked_query,
                "attack_method": attack_method,
            }
        )
    if not csv_rows:
        csv_rows = [
            {
                "queryid": str(q_id),
                "raw_query": raw_query,
                "attacked_query": raw_query,
                "attack_method": config.attack_method,
            }
            for q_id, raw_query in original_queries.items()
        ]
    variations_df = pd.DataFrame(csv_rows)
    attacked_queries_csv_path = os.path.join(save_dir, f"{filename}.csv")
    variations_df.to_csv(attacked_queries_csv_path, index=False)
    print(f"Saved attacked queries to {attacked_queries_csv_path} in csv format")
    return attacked_queries_json_path, attacked_queries_csv_path



def main():

    '''
    New attack code is used to generate attack queries, and the methods are as traceable as possible.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config  =  config_parse()
    config.device = device

    # Add wandb initialization
    wandb.init(
        project="LLM_Robust_Query_Variation_Generation", # You can modify the wandb project name as you want
        config=vars(config),
        name=f"{config.dataset}_{config.attack_method}_{config.split}_seed_{config.seed}"
    )

    set_seed(config.seed)
    output_save_root = DEFAULT_ADV_SCORES_ROOT


    data_output_dic: Dict[str, Any] = load_beir_data(config.dataset ,split=config.split)
    corpus, queries, qrels = data_output_dic['corpus'], data_output_dic['queries'], data_output_dic['qrels']
    queries_raw = data_output_dic['queries_raw'] if 'queries_raw' in data_output_dic else None

    query_ids=list(queries.keys())
    query_texts=list(queries.values())

    def _generate_variations(method: str):
        if method == 'mispelling':
            return MispellingActions(query_texts, query_ids).mispelling_chars()
        if method == 'ordering':
            return OrderingActions(query_texts, query_ids).shuffle_word_order()
        if method == 'synonym':
            return SynonymActions(query_texts, query_ids).adversarial_synonym_replacement()
        if method == 'paraphrase':
            paraphrase_actions = ParaphraseActions(query_texts, query_ids, uqv_model_path=config.uqv_model_path)
            paraphrase_results = paraphrase_actions.back_translation_paraphrase()
            paraphrase_results += paraphrase_actions.seq2seq_paraphrase()
            return paraphrase_results
        if method == 'naturality':
            return NaturalityActions(query_texts, query_ids).remove_stop_words()
            # naturality_result += naturality_actions.naturality_by_trec_desc_to_title() # the model is not available now

        raise NotImplementedError(f"Not implemented for attack_method: {method}")

    variations = _generate_variations(config.attack_method)
    save_attacked_queries(variations, queries, output_save_root, config)

if __name__ == "__main__":
    main()