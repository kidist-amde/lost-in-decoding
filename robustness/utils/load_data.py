import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sympy import im
from transformers import AutoTokenizer,AutoModel
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import DPRQuestionEncoder
from transformers import BertModel
from sentence_transformers import SentenceTransformer
import faiss
import torch
import logging
import os
import requests
import zipfile
from tqdm import tqdm
from data_loader import BeirGenericDataLoader,GenericDataLoader #beir.datasets
# from beir.datasets.data_loader_yk import GenericDataLoader
import random
import numpy as np
from torch.utils.data import DataLoader
import time
from collections import Counter
from transformers.data.data_collator import default_data_collator
from sklearn.cluster import KMeans
# from data_loader import Attack_Batch_Dataset
from typing import TypedDict, Any, Dict

logger = logging.getLogger(__name__)

def download_url(url: str, save_path: str, chunk_size: int = 1024):
    """Download url with progress bar using tqdm
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads

    Args:
        url (str): downloadable url
        save_path (str): local path to save the downloaded file
        chunk_size (int, optional): chunking of files. Defaults to 1024.
    """
    r = requests.get(url, stream=True)
    total = int(r.headers.get('Content-Length', 0))
    with open(save_path, 'wb') as fd, tqdm(
        desc=save_path,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=chunk_size,
    ) as bar:
        for data in r.iter_content(chunk_size=chunk_size):
            size = fd.write(data)
            bar.update(size)

def unzip(zip_file: str, out_dir: str):
    zip_ = zipfile.ZipFile(zip_file, "r")
    zip_.extractall(path=out_dir)
    zip_.close()

def download_and_unzip(url: str, out_dir: str, chunk_size: int = 1024) -> str:
    os.makedirs(out_dir, exist_ok=True)
    dataset = url.split("/")[-1]
    zip_file = os.path.join(out_dir, dataset)

    if not os.path.isfile(zip_file):
        logger.info("Downloading {} ...".format(dataset))
        download_url(url, zip_file, chunk_size)

    if not os.path.isdir(zip_file.replace(".zip", "")):
        logger.info("Unzipping {} ...".format(dataset))
        unzip(zip_file, out_dir)

    return os.path.join(out_dir, dataset.replace(".zip", ""))

def load_beir_data(dataset_name, split):  #Download datasets from extended_beir_datasets project repository
    # Load datasets
    # url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name) # Original BEIR datasets
    url = f"https://github.com/liyongkang123/extended_beir_datasets/releases/download/beir_v1.0/{dataset_name}.zip"
    hf_home = os.getenv('HF_HOME')  # Huggingface datasets cache dir to avoid downloading the dataset for each project
    if hf_home:
        out_dir = os.path.join(hf_home, "datasets")
    else:
        out_dir = os.path.join(os.getcwd(), "datasets")  # If the environment variable is not set, use the current working directory
    print('out_dir: ', out_dir)

    data_path = os.path.join(out_dir, dataset_name)
    if not os.path.exists(data_path):
        data_path = download_and_unzip(url, out_dir)
    print(data_path)

    # bright_data = BeirGenericDataLoader(data_path)
    data = GenericDataLoader(data_path)
    if '-train' in data_path: # Because there are datasets named nq-train
        split = 'train'
    elif split=='test'  and ('msmarco' in dataset_name):  # Because the msmarco dataset does not use test, but dev
        split = ['dev','trec_dl19','trec_dl20',]
    elif split=='test' and 'browsecomp_plus' in dataset_name:
        split = ['golds','evidence']
    # corpus, queries, qrels = data.load(split=split) # corpus is a dict, containing information about all passages, including title, text, id, etc.
    #
    # return corpus, queries, qrels

    output_dic: Dict[str, Any]= data.load(split=split) # corpus is a dict, containing information about all passages, including title, text, id, etc.
    return output_dic


