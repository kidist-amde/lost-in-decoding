"""
Evaluate query-variation and corpus-poisoning attacks on BEIR retrieval results.

Note: This script only supports dense retrieval.
"""

from transformers.trainer_utils import set_seed
import wandb
 
import argparse
from robustness.utils.load_model import load_model_hf
from robustness.utils.load_data import load_beir_data
from robustness.utils.logging import LoggingHandler
from typing import Dict, Any, Optional, List
from pathlib import Path
 
# from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES_GPU
from robustness.utils.beir_exact_search import DenseRetrievalExactSearch as DRES_GPU
from beir.retrieval.search.dense.util import cos_sim, dot_score, pickle_load
from robustness.utils.beir_utils import (
    DenseEncoderModel,
    SentenceEncoderModel,
    SentenceEncoderModel_Prompt,
)
from robustness.utils.utils import (
    replace_slash,
    get_last_element,
    merge_beir_eval_scores,
    to_numpy,
    bright_scores_remove_excluded_ids,
    save_embeddings,
)
 
import logging
import os
import json
import torch
import sys
import transformers
from torch import nn
 
from robustness.utils.utils import get_model_prompts_tasks
 

 
from robustness.utils.utils import calculate_retrieval_metrics
from tqdm import tqdm
import glob
from typing import TypedDict, Any, Dict
#### Just some code to print debug information to stdout
# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)
#### /print debug information to stdout


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROBUSTNESS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def remove_identical_ids(results):
    popped = []
    for qid, rels in results.items():
        for pid in list(rels):
            if qid == pid:
                results[qid].pop(pid)
                popped.append(pid)
    return results

# Configure constants
class Config_Adv:
    # config for adv test
    BRIGHT_DATASETS = {
        "biology", "earth_science", "economics", "psychology", "robotics",
        "stackoverflow", "sustainable_living", "leetcode", "pony", "aops",
        "theoremqa_theorems", "theoremqa_questions"
    }
    BEIR_DATASETS = {
    "trec-covid",    "nfcorpus", "nq", "hotpotqa", "fiqa",  "arguana",
    "webis-touche2020",    "quora", "dbpedia-entity", "scidocs", "fever",
    "climate-fever",     "scifact",
    }

    MSMARCO_SPLITS = ['dev', 'trec_dl19', 'trec_dl20']
    BROWSECOMP_SPLITS = ['golds', 'evidence']

    DEFAULT_K_VALUES = [1, 5, 10, 50, 100, 1000]
    DEFAULT_CLEAN_EMBEDDING_ROOT = './embeddings/clean'
    DEFAULT_CLEAN_SCORES_ROOT = os.path.join(ROBUSTNESS_ROOT, "output")

    DEFAULT_ADV_EMBEDDING_ROOT = './embeddings/adversarial'
    DEFAULT_ADV_SCORES_ROOT = os.path.join(ROBUSTNESS_ROOT, "output_attack")

    WANDB_PROJECT = "LLM_robust_eval_attack"


PENHA_ATTACK_METHODS = {"mispelling", "ordering", "synonym", "paraphrase", "naturality"}
PENHA_ATTACK_METHOD_ALIASES = {
    # Backward compatibility for the previous generic "typo" flag
    "typo": "mispelling",
}


def _resolve_attack_method(attack_method: str) -> str:
    """Map deprecated attack names to concrete Penha methods."""
    return PENHA_ATTACK_METHOD_ALIASES.get(attack_method, attack_method)


def get_attacked_queries_path(config, root=None, attack_method: Optional[str] = None):
    """
    Build the file path for pre-generated attacked queries.
    """
    method_name = attack_method or config.attack_method
    root = root or Config_Adv.DEFAULT_ADV_SCORES_ROOT
    filename = (
        f"{config.dataset}_{config.split}_attacked_queries_seed_{config.seed}"
        f"_attack_method_{method_name}.json"
    )
    dataset_dir = os.path.join(root, "attacked_text", "query", config.dataset)
    return os.path.join(dataset_dir, filename)


def load_attacked_queries(config, root=None, attack_method: Optional[str] = None):
    """
    Load attacked queries JSON produced by generate_penha.py.
    """
    attacked_queries_path = get_attacked_queries_path(config, root, attack_method=attack_method)
    if not os.path.exists(attacked_queries_path):
        raise FileNotFoundError(
            f"Attacked queries file not found: {attacked_queries_path}. "
            "Please run augment_texts.py beforehand."
        )

    with open(attacked_queries_path, "r", encoding="utf-8") as f:
        return json.load(f), attacked_queries_path

def get_poisoning_docs_path(config):
    if config.dataset =='nq':
        target_dataset = 'nq-train'
    else:
        target_dataset = config.dataset
    
    if config.model_name == 'contriever':
        target_retrieval_model = 'contriever-msmarco'
    else:
        target_retrieval_model = config.model_name
    
    output_dir = Path(ROBUSTNESS_ROOT) / "output_attack" / "attacked_text" / "document" / target_dataset
    # Build the file name containing the hyperparameters, for easy identification of experimental configurations
    config.adv_tokens_num = getattr(config, 'adv_tokens_num', 50)
    config.attack_iterations = getattr(config, 'attack_iterations', 5000)
    config.candidate_pool_size = getattr(config, 'candidate_pool_size', 100)
    config.init_method = getattr(config, 'init_method', True)

    if config.seed == 'all':
        seeds_list = [1999, 5, 27, 2016, 2026]
    else:
        seeds_list = [config.seed]

    file_name_list = []
    for seed in seeds_list:
        file_name = (
            f"{target_dataset}_train_attacked_documents_"
            f"seed_{seed}_"
            f"{target_retrieval_model}_supervised_hotflip_"
            f"N{config.attacked_num}_"  # Attack document number
            f"Len{config.adv_tokens_num}_"  # Adversarial Token length
            f"Iter{config.attack_iterations}_"  # Iteration times
            f"Pool{config.candidate_pool_size}_"  # Candidate pool size
            f"Init{config.init_method}" # Initialization method
            f".json"
        )

        file_name_list.append(os.path.join(output_dir, file_name))
    return file_name_list

def load_poisoning_docs(config,tokenizer):
    poisoning_docs_path_list = get_poisoning_docs_path(config)
    adversarial_documents_dict = {}

    for idx, poisoning_docs_path in enumerate(poisoning_docs_path_list):
        if not os.path.exists(poisoning_docs_path):
            raise FileNotFoundError(
                f"Poisoning docs file not found: {poisoning_docs_path}. "
                "Please run attack_documents_supervised.py beforehand."
            )
        with open(poisoning_docs_path, "r", encoding="utf-8") as f:
            poisoning_docs = json.load(f)
        # Check if the number of poisoning_docs in the file is correct
        assert len(poisoning_docs) == config.attacked_num
            
        # Read the token_ids of the document, and convert it to the text of the tokenizer
        for key, value in poisoning_docs.items():
            token_ids = value['best_adv_passage_ids']
            adv_text = tokenizer.decode(token_ids, skip_special_tokens=False)
            # print(f"adv_text: {adv_text}")
            adversarial_documents_dict[f"adv_doc_{idx}_{key}"] = {"title": "", "text": adv_text,'token_ids': token_ids}
    print('----------------------------------------------------------------')
    print('adversarial_documents_dict length:', {len(adversarial_documents_dict)})
    return adversarial_documents_dict, poisoning_docs_path_list

 

def config_parse():
    # Parse command line arguments for evaluation
    parser = argparse.ArgumentParser(description="Evaluate BEIR retrieval results with different LLM-based IR models.")
    parser.add_argument("--dataset", type=str, default="nq", help="Dataset to evaluate on (e.g., 'nq', 'msmarco' arguana nfcorpus scifact scidocs fiqa trec-covid)")
    parser.add_argument("--model_name", type=str, default="linq", help="Path to the evaluation model. bge_reasoner reasonir gte qwen3 linq contriever bge_m3  diver有问题其他的可以正常运行")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use for evaluation (default: 'test').")
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=64, help="Batch size for evaluation (default: 256).")
    parser.add_argument('--seed', type=str, default='1999', help='Seed for evaluation. 1999 5 27 2016 2026 all')
    # parser.add_argument("--embedding_root", type=str, default=Config.DEFAULT_EMBEDDING_ROOT,help="Embedding save root path"    )
    # parser.add_argument("--scores_root", type=str, default=Config.DEFAULT_SCORES_ROOT,help="Scores save root path")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging" )
    parser.add_argument(
        "--attack_method",
        type=str,
        default="none",
        help="Attack method for query corruption (mispelling, ordering, synonym, paraphrase, naturality, supervised_poisoning   ,none).",
    )
    parser.add_argument(
        "--attacked_num",
        type=int,
        default=50,
        help="Number of attacked documents. 10 or 50.",
    )

    args = parser.parse_args()
    return args

def setup_experiment(config: argparse.Namespace) -> Dict[str, Any]:
    """Setup experiment environment and path"""
    if config.seed == 'all':
        set_seed(1999)
    else:
        set_seed(int(config.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    # Path setting
    adv_scores_save_root = Config_Adv.DEFAULT_ADV_SCORES_ROOT
    adv_embedding_save_path = os.path.join(
        Config_Adv.DEFAULT_ADV_EMBEDDING_ROOT,
        f'output/embeddings/{config.model_name}/{config.dataset}/{config.attack_method}_seed_{config.seed}_attacked_num_{config.attacked_num}'
    )
    os.makedirs(adv_embedding_save_path, exist_ok=True)

    # Clean existing path
    clean_embedding_save_path = os.path.join(
        Config_Adv.DEFAULT_CLEAN_EMBEDDING_ROOT,
        f'output/embeddings/{config.model_name}/{config.dataset}'
    )
    clean_scores_save_root = Config_Adv.DEFAULT_CLEAN_SCORES_ROOT

    # Initialize wandb
    wandb_run = None
    if not config.no_wandb:
        wandb_run = wandb.init(
            project= Config_Adv.WANDB_PROJECT,
            config=vars(config),
            name=f"{config.model_name}_{config.dataset}_{config.split}"
        )

    logger.info(f"Configuration: {config}")

    return {
        'device': device,
        'adv_embedding_save_path': adv_embedding_save_path,
        'adv_scores_save_root': adv_scores_save_root,
        'clean_embedding_save_path': clean_embedding_save_path,
        'clean_scores_save_root': clean_scores_save_root,
        'wandb_run': wandb_run
    }

def attack(config, attack_method, model,experiment_setup,tokenizer,prompts):
    # Given clean corpus and query, output the corpus embedding and query embedding after attack
    # Here target_corpus and target_query are dict format
    use_token_type_ids = hasattr(tokenizer, "model_input_names") and "token_type_ids" in tokenizer.model_input_names

    resolved_attack_method = _resolve_attack_method(attack_method)

    if resolved_attack_method in PENHA_ATTACK_METHODS:
        # Attack the query, the document remains unchanged
        adversarial_queries, attacked_queries_path = load_attacked_queries(
            config,
            experiment_setup['adv_scores_save_root'],
            attack_method=resolved_attack_method,
        )
        print(f"Loaded attacked queries from {attacked_queries_path}")
        # Convert the query to embeddings and save
        # encode queries
        updated_queries_embeddings = model.model.encode_queries(
            [adversarial_queries[qid] for qid in adversarial_queries], # The input is a list
            batch_size= config.per_gpu_eval_batch_size,
        )
        query_ids = list(adversarial_queries.keys())
        # Save embedding
        adversarial_query_embeddings_file_path = os.path.join(experiment_setup['adv_embedding_save_path'], "adversarial_query_embeddings.pkl")
        print(f"Saving query embeddings to {adversarial_query_embeddings_file_path}")
        save_embeddings(updated_queries_embeddings, query_ids, adversarial_query_embeddings_file_path)
        clean_corpus_embeddings_files_path = glob.glob(os.path.join(experiment_setup['clean_embedding_save_path'], "corpus.*.pkl"))
        adversarial_corpus_embeddings_files_path = clean_corpus_embeddings_files_path

    elif resolved_attack_method == "none":
        # # No attack, just to check if the code is correct
        clean_corpus_embeddings_files_path = glob.glob(os.path.join(experiment_setup['clean_embedding_save_path'], "corpus.*.pkl"))
        clean_query_embeddings_file_path =  os.path.join(experiment_setup['clean_embedding_save_path'], "queries.pkl")

        adversarial_query_embeddings_file_path = clean_query_embeddings_file_path
        adversarial_corpus_embeddings_files_path = clean_corpus_embeddings_files_path
    
    elif resolved_attack_method == "supervised_poisoning":
        # Use supervised poisoning attack, the query remains unchanged, and the document increases the poisoning document
        
        # Load the poisoning document
        adversarial_docs_dict, _ = load_poisoning_docs(config,tokenizer)
        adversarial_docs_texts = [adversarial_docs_dict[doc]['text'] for doc in adversarial_docs_dict.keys()]
        print(f"adversarial_docs_texts length: {len(adversarial_docs_texts)}")
        # 2. Encode the new documents, batch by batch

        passage_prefix = prompts.get("passage", "")
        prompt_tokens = tokenizer(passage_prefix, padding=False, add_special_tokens=False, return_tensors='pt').input_ids[0].tolist()
        print(f"prompt_tokens: {prompt_tokens}")
        adv_ds = [prompt_tokens + doc['token_ids'] for doc in adversarial_docs_dict.values()]
        # Convert the passage_prefix to token_ids, and then add it to the front of adv_ds

        # Batch process embedding, to avoid GPU memory insufficient
        batch_size = config.per_gpu_eval_batch_size
        num_samples = len(adv_ds)
        all_embeddings = []
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_ds = adv_ds[start_idx:end_idx]
            
            adv_p_ids = torch.tensor(batch_ds).cuda()
            adv_attention = torch.ones_like(adv_p_ids, device='cuda')

            if use_token_type_ids:
                adv_token_type = torch.zeros_like(adv_p_ids, device='cuda')
                adv_input = {'input_ids': adv_p_ids, 'attention_mask': adv_attention, 'token_type_ids': adv_token_type}
            else:
                adv_input = {'input_ids': adv_p_ids, 'attention_mask': adv_attention}
            
            with torch.no_grad():
                adv_embeddings = model.model.encoder.hf_model(**adv_input).last_hidden_state
                batch_embeddings = model.model.encoder._pooling(adv_embeddings, adv_attention, prompt=passage_prefix)
                all_embeddings.append(batch_embeddings.cpu())
        adversarial_docs_embeddings = torch.cat(all_embeddings, dim=0)  # Concatenate all batch embeddings


        # Do you need to save the document embedding of adv?
        adversarial_document_embeddings_file_path = os.path.join(experiment_setup['adv_embedding_save_path'], "corpus.adv_docs.pkl")
        print(f"Saving document embeddings to {adversarial_document_embeddings_file_path}")
        save_embeddings(adversarial_docs_embeddings, list(adversarial_docs_dict.keys()), adversarial_document_embeddings_file_path)


        # The embedding file path of the query is clean, because the query remains unchanged
        clean_corpus_embeddings_files_path = glob.glob(os.path.join(experiment_setup['clean_embedding_save_path'], "corpus.*.pkl"))
        clean_query_embeddings_file_path =  os.path.join(experiment_setup['clean_embedding_save_path'], "queries.pkl")
        adversarial_corpus_embeddings_files_path = clean_corpus_embeddings_files_path + [adversarial_document_embeddings_file_path]
        adversarial_query_embeddings_file_path = clean_query_embeddings_file_path
    else:
        adversarial_query_embeddings_file_path = None
        adversarial_corpus_embeddings_files_path = None
        raise ValueError(f"Attack method {config.attack_method} is not supported")

    return adversarial_query_embeddings_file_path, adversarial_corpus_embeddings_files_path


def DenseRetrieval(config):
    # Setup experiment environment
    experiment_setup = setup_experiment(config)

    #### load eval_dataset
    # split = 'test' or 'dev', default is test
    data_output_dic: Dict[str, Any] = load_beir_data(config.dataset ,split=config.split)
    corpus, queries, qrels = data_output_dic['corpus'], data_output_dic['queries'], data_output_dic['qrels']
    queries_raw = data_output_dic['queries_raw'] if 'queries_raw' in data_output_dic else None

    #### Load the   model and retrieve
    prompts = get_model_prompts_tasks(model_name=config.model_name,dataset_name=config.dataset)
    print(f"prompts: {prompts}")

    encoder,tokenizer = load_model_hf(config.model_name)
 
    # For large datasets, when using GPU, it is necessary to ensure that there are at least 2 GPUs, otherwise the memory will be insufficient and an error will be reported, here uses the custom DRES_GPU
    model = DRES_GPU(SentenceEncoderModel_Prompt(encoder, prompts, config.model_name), batch_size=config.per_gpu_eval_batch_size)

    # retriever = EvaluateRetrieval(model, score_function=score_function, k_values=[1, 5, 10, 50 ,100, 1000] , ) # or "dot" for dot product cos_sim

    adversarial_query_embeddings_file_path, corpus_embeddings_files_path = attack(config, config.attack_method, model, experiment_setup,tokenizer,prompts)
    score_function = "cos_sim" if config.model_name != 'contriever' else "dot"
    results = model.search_from_files( # Here the search_from_files is my own DRES_GPU, added to select whether to perform l2-normalize according to score_function
        query_embeddings_file=adversarial_query_embeddings_file_path,
        corpus_embeddings_files=corpus_embeddings_files_path, # Here is a list, containing clean corpus embedding and adversarial corpus embedding
        top_k=1000,
        score_function= score_function,
    )

 

    # Process special datasets, if the dataset is bright, then excluded_ids need to be removed from results # In fact, non-StackExchange 5 datasets need to do this
    if config.dataset in Config_Adv.BRIGHT_DATASETS :
        results = bright_scores_remove_excluded_ids(queries_raw, results)
    
    if config.dataset == 'arguana':
        results = remove_identical_ids(results)

    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]

    if 'msmarco' in config.dataset:
        splits = ['dev' ,'trec_dl19' ,'trec_dl20' ,]
    elif 'browsecomp_plus' in config.dataset:
        splits = ['golds','evidence']
    else:
        splits = [config.split]
        qrels = {config.split: qrels}

    for split_name in splits:
        split_qrels = qrels[split_name]
        split_results = {qid: docs for qid, docs in results.items() if qid in split_qrels} # Only keep results in split_qrels
        _evaluate_split(split_name, split_results, split_qrels, config, experiment_setup['adv_scores_save_root'], experiment_setup['wandb_run'])


def _evaluate_split(split, results, split_qrels, config, scores_save_root, wandb_logger=None):
    """Evaluate single data split"""
    scores_save_path = os.path.join(
        scores_save_root, f'scores/{config.model_name}/{config.dataset}_{split}/attack_method_{config.attack_method}_seed_{config.seed}_attacked_num_{config.attacked_num}'
    )
    os.makedirs(scores_save_path, exist_ok=True)

    print(f"--------------------------------split: {split}--------------------------------")

    # Calculate retrieval metrics
    output_all_score, merged_query_level_scores = calculate_retrieval_metrics(
        results=results, qrels=split_qrels, return_scores=True
    )

    asr_scores = calculate_attack_success_rate(results, k=[5, 10, 20, 50], prefix="adv_doc")
    print(f"Attack Success Rates for split {split}:")
    for k_name, asr_value in asr_scores.items():
        print(f"  {k_name}: {asr_value:.4f}")
    # Save results
    with open(f'{scores_save_path}/merged_scores.json', 'w') as f:
        json.dump(merged_query_level_scores, f)
    with open(f'{scores_save_path}/retrieval_results.json', 'w') as f:
        json.dump(results, f)

    # Save output_all_score and asr_scores as a json file
    with open(f'{scores_save_path}/metrics_scores_and_asr.json', 'w') as f:
        json.dump({**output_all_score, **asr_scores}, f)
 

    # Record to wandb (if logger is provided)
    if wandb_logger:
        log_dict = {
            f"{split}_NDCG@10": output_all_score['NDCG@10'],
            f"{split}_Recall@100": output_all_score['Recall@100'],
            f"{split}_MRR@10": output_all_score['MRR@10'],
        }
        # Add all ASR metrics
        for k_name, asr_value in asr_scores.items():
            log_dict[f"{split}_{k_name}"] = asr_value
        wandb_logger.log(log_dict)

def calculate_attack_success_rate(results, k=[5, 10, 20, 50], prefix="adv_doc"):
    """
    Calculate attack success rate (ASR@k)
    As long as the document starting with prefix is included in the Top-K results, it is considered to be successful.
    
    Args:
        results: Retrieval result dictionary {query_id: {doc_id: score, ...}, ...}
        k: Integer list, e.g. k=[5, 10, 20, 50]
        prefix: The prefix identifier of the attacked document
    
    Returns:
        Dictionary {f"ASR@{k_val}": asr_value, ...}
    """
    total_queries = len(results)
    
    if total_queries == 0:
        return {f"ASR@{k_val}": 0.0 for k_val in k}
    
    k_list = sorted(k)  # Sort to optimize calculation
    max_k = max(k_list)
    success_counts = {k_val: 0 for k_val in k_list}
    
    for query_id, doc_scores in results.items():
        # Sort only once, take the largest k
        sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:max_k]
        
        # Check if the attack is successful for each k value
        for k_val in k_list:
            top_k_docs = sorted_docs[:k_val]
            if any(doc_id.startswith(prefix) for doc_id, score in top_k_docs):
                success_counts[k_val] += 1
    
    return {f"ASR@{k_val}": success_counts[k_val] / total_queries for k_val in k_list}

# Save the retrieval results
if __name__ == '__main__':
    config = config_parse()
    DenseRetrieval(config)
