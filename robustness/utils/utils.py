import json
import os
import torch
import pickle
from typing import Dict, List, Optional, Sequence, Tuple,Mapping

# Replace / in string with _
def replace_slash(s):
    return s.replace("/", "_")

# Split string by / and return the last element
def get_last_element(s):
    return s.split("/")[-1]

def move_to_device(sample, device: torch.device):
    if len(sample) == 0:
        return {}

    def _move_to_device(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device, non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_device(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_device(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_device(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            # For Mapping type, return dictionary directly
            return {k: _move_to_device(v) for k, v in maybe_tensor.items()}
        else:
            return maybe_tensor

    return _move_to_device(sample)

# def get_model_name(model_path):
#     model_name_list = model_path.split("/")
#     model_id = "_".join(model_name_list[2:])
#     return model_id

def merge_beir_eval_scores(*score_dicts):
    """
    Merge multiple per-query scores dicts into a single dict

    Args:
        *score_dicts: Input multiple dicts, each is {qid: {metric: value}}

    Returns:
        merged_scores: {qid: {metric1: value1, metric2: value2, ...}}
    """
    merged_scores = {}

    for scores in score_dicts:
        for qid, metrics in scores.items():
            if qid not in merged_scores:
                merged_scores[qid] = {}
            merged_scores[qid].update(metrics)
    return merged_scores

import numpy as np
def to_numpy(x):
    # Case 1: Directly a Tensor (1D/2D can both be supported)
    if isinstance(x, torch.Tensor):
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.to(torch.float32)  # Convert to float32 to avoid some numpy ops not supported later
        return x.detach().cpu().numpy()

    # Case 2: Is list / tuple, possibly containing 1D Tensor
    if isinstance(x, (list, tuple)):
        # Recursively convert all elements to numpy
        converted = [to_numpy(e) for e in x]

        # If all are ndarray, try to stack them into a matrix
        if all(isinstance(e, np.ndarray) for e in converted):
            try:
                # For example, a bunch of vectors with shape=(D,) → shape=(N, D)
                return np.stack(converted, axis=0)
            except ValueError:
                # If the shapes don't match, return list[np.ndarray] to let the upper layer decide how to handle
                return converted
        return converted

    # Case 3: Can add some defensive support, like dict
    if isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}

    # Other types: numpy arrays, floats, ints, etc. directly return
    return x


def get_model_prompts_tasks(model_name, dataset_name):
    '''

    :param model_name:
    :param dataset_name:
    :return: Dict{query : query_task, passage: doc_task}
    '''
    Dataset_NAME_ALIASES = {
        "nq-train": "nq",
    }
    dataset_name = Dataset_NAME_ALIASES.get(dataset_name, dataset_name)
    # Get parent directory path (parent directory of the path that calls utils.py)
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Build the correct prompts file path
    prompts_path = os.path.join(parent_dir, 'prompts', f'{model_name}.json')

    # Check if the prompts file exists
    if not os.path.isfile(prompts_path):
        raise FileNotFoundError(
            f"Prompt file not found at expected location: {prompts_path}\n"
            "Ensure the file exists and the model_name is correct."
        )
    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts_all = json.load(f)
    # Check if dataset_name exists in the JSON file
    if dataset_name not in prompts_all:
        raise KeyError(
            f"Dataset '{dataset_name}' not found in prompts file: {prompts_path}\n"
            "Ensure the dataset exists in the provided file."
        )

    prompts = prompts_all[dataset_name]

    if  model_name in  ['qwen3', 'qwen3_4B', 'qwen3_0.6B', 'linq', 'diver', 'diver_1.7B', 'diver_0.6B', 'gte', 'bge_reasoner'] : # qwen3 requires adding this prompt only to queries, not documents
        prompts['query'] = f"Instruct: {prompts['query']}\nQuery:"
        # reasonir and bge_m3 and contriever do not need to

    elif dataset_name=='browsecomp_plus':
        prompts['query'] = f"Instruct: {prompts['query']}\nQuery:"

    return  prompts


def bright_scores_remove_excluded_ids(queries_raw, scores):
    '''
    eval bright datasets, we need to remove excluded ids from scores after retreival
    :param query:
    :param scores:
    :return:
    '''
    assert isinstance(queries_raw, list)
    for query in queries_raw:
            excluded_ids = query['metadata']['excluded_ids']
            for excluded_id in set(excluded_ids):
                if excluded_id !="N/A":
                    if excluded_id in scores[query['_id']]:
                        scores[query['_id']].pop(excluded_id)
    return scores


def calculate_retrieval_metrics(results, qrels, k_values=[1, 5, 10, 25, 50, 100,1000], return_scores=False):
    import pytrec_eval
    from beir.retrieval.custom_metrics_yk import mrr as mrr_func

    '''
    This function is copied from the resonir code, mainly using pytrec_eval to calculate ndcg map recall precision etc.
    I also added the calculation of MRR@k in this function, refer to the mrr function in custom_metrics_yk.py
    This function returns two things:
    1. output: A dictionary containing the average values of ndcg map recall precision mrr etc.
    2. final_scores: A dictionary containing the scores of each query's various indicators, convenient for subsequent analysis
    For example:
    '''

    # https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/evaluation.py#L66
    # follow evaluation from BEIR, which is just using the trec eval
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    mrr = {"MRR": 0.0} # Here it is not supported to calculate mrr@k, only the overall MRR is calculated

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    # https://github.com/cvangysel/pytrec_eval/blob/master/examples/simple_cut.py
    # qrels = {qid: {'pid': [0/1] (relevance label)}}
    # results = {qid: {'pid': float (retriever score)}}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string, "recip_rank"})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]
        mrr["MRR"] += scores[query_id]["recip_rank"]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)
    mrr["MRR"] = round(mrr["MRR"] / len(scores), 5)

    # oracle reranker evaluation
    sorted_ids = {}
    top_100_ids = {}
    for query_id in results.keys():
        sorted_ids[query_id] = sorted(results[query_id].keys(), key=lambda x: results[query_id][x], reverse=True)
        top_100_ids[query_id] = set(sorted_ids[query_id][:100])
    oracle_results = {}
    for query_id in results.keys():
        oracle_results[query_id] = {}
        for doc_id in results[query_id].keys():
            if doc_id in top_100_ids[query_id] and query_id in qrels and doc_id in qrels[query_id]: # a doc is both top 100 and also in ground truth
                oracle_results[query_id][doc_id] = qrels[query_id][doc_id] # extract the score from ground truth
            else:
                oracle_results[query_id][doc_id] = 0
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string, "recip_rank"})
    oracle_scores = evaluator.evaluate(oracle_results) # The indicator values in oracle_scores will overwrite the same indicator values in scores. So be careful to distinguish and do not merge temporarily (not merge yet)
    oracle_ndcg = {}
    for k in k_values:
        oracle_ndcg[f"Oracle NDCG@{k}"] = 0.0
    for query_id in oracle_scores.keys():
        for k in k_values:
            oracle_ndcg[f"Oracle NDCG@{k}"] += oracle_scores[query_id]["ndcg_cut_" + str(k)]
    for k in k_values:
        oracle_ndcg[f"Oracle NDCG@{k}"] = round(oracle_ndcg[f"Oracle NDCG@{k}"] / len(oracle_scores), 5)


    if return_scores:
        # This is my own addition, I added the calculation of MRR@k in this function
        MRR_cut_dict, mrr_per_query_scores = mrr_func(qrels, results, k_values, return_scores=return_scores)
        final_scores = merge_beir_eval_scores(scores, mrr_per_query_scores) # query-level scores
        output = {**ndcg, **_map, **recall, **precision, **mrr, **oracle_ndcg, **MRR_cut_dict}
        print(output)
        return output, final_scores
    else:
        MRR_cut_dict = mrr_func(qrels, results, k_values, return_scores=return_scores)
        output = {**ndcg, **_map, **recall, **precision, **mrr, **oracle_ndcg, **MRR_cut_dict}
        print(output)
        return output



def save_embeddings(
    embeddings: np.ndarray | list[torch.Tensor], text_ids: list[str], output_filename: str = "./embeddings/"
): # The save_embeddings function in beir is basically the same, the only difference is that here the embeddings are forcibly converted to float32 to use the model output of bfloat16
    """
    Saves the embeddings to a pickle file.
    :param embeddings: The embeddings to save.
    :param output_path: The path where the embeddings will be saved.
    """
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    if isinstance(embeddings[0], torch.Tensor):
        embeddings = embeddings.float().cpu().detach().numpy()  # Convert to numpy array if it's a tensor

    with open(output_filename, "wb") as f:
        pickle.dump((embeddings, text_ids), f)


def build_prefix_suffix_tokens(tokenizer, doc_prefix: str) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    if tokenizer is None:
        return None, None
    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    # --- Correction part: Handle start Token (CLS or BOS) ---
    # Use CLS first, if not, use BOS (common in LLama/Qwen/GPT)
    start_token_id = tokenizer.cls_token_id
    if start_token_id is None:
        start_token_id = tokenizer.bos_token_id
    
    # Note: Some tokenizers (such as RoBERTa) cls_token_id and bos_token_id are the same
    # As long as a valid start token is found, add it
    if start_token_id is not None:
        prefix_tokens.append(start_token_id)

    # Process doc_prefix text
    if doc_prefix:
        # add_special_tokens=False is correct, to prevent the insertion of extra CLS/SEP in the middle
        prefix_ids = tokenizer(doc_prefix, add_special_tokens=False).input_ids
        prefix_tokens.extend(prefix_ids)

    # --- Correction part: Handle end Token (SEP or EOS) ---
    # Use SEP first, if not, use EOS
    end_token_id = tokenizer.sep_token_id
    if end_token_id is None:
        end_token_id = tokenizer.eos_token_id
        
    if end_token_id is not None:
        suffix_tokens.append(end_token_id)

 
    return prefix_tokens, suffix_tokens