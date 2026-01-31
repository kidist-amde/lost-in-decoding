# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from collections import defaultdict
from typing import List, Dict, Optional
import numpy as np
import torch
import torch.distributed as dist
import glob
from tqdm import tqdm

import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch

from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank

from robustness.utils import dist_utils, normalize_text
from robustness.utils.utils import to_numpy

class DenseEncoderModel:
    def __init__(
        self,
        query_encoder,
        doc_encoder,
        tokenizer,
        q_max_len=512, # 两个默认全搞成512
        p_max_len=512,
        sep: str = " ",
        prompts: dict[str, str] = None,
        add_special_tokens=True,
        norm_query=False,
        norm_doc=False,
        lower_case=False,
        # normalize_text=False,
        **kwargs,
    ):
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        self.query_encoder.eval()
        self.doc_encoder.eval()

        self.tokenizer = tokenizer
        self.q_max_len = q_max_len
        self.p_max_len = p_max_len
        self.add_special_tokens = add_special_tokens
        self.norm_query = norm_query
        self.norm_doc = norm_doc
        self.lower_case = lower_case
        # self.normalize_text = normalize_text
        self.sep = sep
        if prompts:
            self.query_prefix = prompts.get("query", "")
            self.doc_prefix = prompts.get("passage", "")
        else:
            self.query_prefix = ""
            self.doc_prefix = ""

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Padding side should be right for LLM-based models such as LLAMA-2 to avoid warnings
        if self.tokenizer.padding_side != "right":
            print(f"Default padding side is {self.tokenizer.padding_side}, making padding side right.")
            self.tokenizer.padding_side = "right"

    def tokenize_text(self, text: str, max_length=512, add_special_tokens=True, return_tensors="pt"):
        batch_dict = self.tokenizer.batch_encode_plus(
            text,
            max_length=max_length,
            padding=True,
            truncation=True,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
        )
        # If the model is DistillBERT, then token_type_ids need to be removed
        if self.doc_encoder.config.model_type == "distilbert":
            del batch_dict["token_type_ids"]
        return batch_dict

    def encode_queries(self, queries: List[str], batch_size: Optional[int] = None, **kwargs) -> np.ndarray:

        # Explicit parameters take precedence, then get from kwargs, then give default values
        if batch_size is None:
            batch_size = kwargs.pop("batch_size", 128)  # Can be replaced with self.default_batch_size

        if dist.is_initialized():
            idx = np.array_split(range(len(queries)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(queries))

        queries = [queries[i] for i in idx]
        # if self.normalize_text:
        #     queries = [normalize_text.normalize(q) for q in queries]
        if self.lower_case:
            queries = [q.lower() for q in queries]

        allemb = []
        nbatch = (len(queries) - 1) // batch_size + 1
        with torch.no_grad():
            # Single card environment or main process display progress bar
            if not dist.is_initialized() or dist.get_rank() == 0:
                iterator = tqdm(range(nbatch), desc="Encoding Queries")
            else:
                iterator = range(nbatch)
            
            for k in iterator:
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(queries))
                sub_texts = [self.query_prefix + text for text in queries[start_idx:end_idx]]
                qencode = self.tokenize_text(sub_texts, max_length=self.q_max_len, add_special_tokens=self.add_special_tokens, return_tensors="pt")
                qencode = {key: value.cuda() for key, value in qencode.items()}
                emb = self.query_encoder.encode_query(qencode)
                allemb.append(emb.cpu())

        allemb = torch.cat(allemb, dim=0)
        allemb = allemb.cuda()
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        return allemb

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: Optional[int] = None, **kwargs):
        # Explicit parameters take precedence, then get from kwargs, then give default values
        if batch_size is None:
            batch_size = kwargs.pop("batch_size", 128)  # Can be replaced with self.default_batch_size

        if dist.is_initialized():
            idx = np.array_split(range(len(corpus)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(corpus))
        corpus = [corpus[i] for i in idx] # Each element is a dictionary, containing "title" and "text"
        corpus_texts = [c["title"] + self.sep + c["text"] if len(c["title"]) > 0 else c["text"] for c in corpus]
        # if self.normalize_text:
        #     corpus_texts = [normalize_text.normalize(c) for c in corpus_texts]
        if self.lower_case:
            corpus_texts = [c.lower() for c in corpus_texts]

        allemb = []
        nbatch = (len(corpus_texts) - 1) // batch_size + 1
        with torch.no_grad():
            # Single card environment or main process display progress bar
            if not dist.is_initialized() or dist.get_rank() == 0:
                iterator = tqdm(range(nbatch), desc="Encoding Queries")
            else:
                iterator = range(nbatch)
            
            for k in iterator:
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(corpus_texts))
                sub_texts = [self.doc_prefix + text for text in corpus_texts[start_idx:end_idx]]
                cencode = self.tokenize_text(sub_texts, max_length=self.p_max_len, add_special_tokens=self.add_special_tokens, return_tensors="pt")
                cencode = {key: value.cuda() for key, value in cencode.items()}
                emb = self.doc_encoder.encode_passage(cencode)
                allemb.append(emb.cpu())

        allemb = torch.cat(allemb, dim=0)
        allemb = allemb.cuda()
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        return allemb


class SentenceEncoderModel:
    '''
    The input model must be a SentenceTransformer model, and it must have the encoder.encode function
    '''
    def __init__(self, encoder, prompts: dict[str, str] = None,  ):
        self.encoder = encoder
        self.sep=" "
        if prompts:
            self.query_prefix = prompts.get("query", "")
            self.doc_prefix = prompts.get("passage", "")
        else:
            self.query_prefix = ""
            self.doc_prefix = ""
 

    def encode_queries(self, queries: List[str], batch_size: Optional[int] = None, **kwargs) -> np.ndarray:

        # Explicit parameters take precedence, then get from kwargs, then give default values
        if batch_size is None:
            batch_size = kwargs.pop("batch_size", 16)  # Can be replaced with self.default_batch_size

        if dist.is_initialized():
            idx = np.array_split(range(len(queries)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(queries))

        queries = [queries[i] for i in idx]
        # if self.normalize_text:
        #     queries = [normalize_text.normalize(q) for q in queries]
        # if self.lower_case:
        #     queries = [q.lower() for q in queries]

        allemb = []
        nbatch = (len(queries) - 1) // batch_size + 1
        with torch.no_grad():
            # Single card environment or main process display progress bar
            if not dist.is_initialized() or dist.get_rank() == 0:
                iterator = tqdm(range(nbatch), desc="Encoding Queries")
            else:
                iterator = range(nbatch)
            
            for k in iterator:
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(queries))
                sub_texts = [self.query_prefix + text for text in queries[start_idx:end_idx]]
                sub_texts = [text if text.strip() else "[PAD]" for text in sub_texts] # Some texts are empty strings, which will cause Diver to report an error
                emb = self.encoder.encode(sub_texts )
                emb = to_numpy(emb)  # Convert all to numpy
                allemb.append(emb)

                # qencode = self.tokenize_text(sub_texts, max_length=self.q_max_len, add_special_tokens=self.add_special_tokens, return_tensors="pt")
                # qencode = {key: value.cuda() for key, value in qencode.items()}
                # emb = self.query_encoder.encode_query(qencode)
                # allemb.append(emb.cpu())
        allemb = np.vstack(allemb)
        # allemb = torch.cat(allemb, dim=0)
        # allemb = allemb.cuda()
        # if dist.is_initialized():
        #     allemb = dist_utils.varsize_gather_nograd(allemb)
        # allemb = allemb.cpu().numpy()
        return allemb

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: Optional[int] = None, **kwargs):
        # Explicit parameters take precedence, then get from kwargs, then give default values
        if batch_size is None:
            batch_size = kwargs.pop("batch_size", 128)  # Can be replaced with self.default_batch_size

        if dist.is_initialized():
            idx = np.array_split(range(len(corpus)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(corpus))
        corpus = [corpus[i] for i in idx] # Each element is a dictionary, containing "title" and "text"
        corpus_texts = [c["title"] + self.sep + c["text"] if len(c["title"]) > 0 else c["text"] for c in corpus]
        # if self.normalize_text:
        #     corpus_texts = [normalize_text.normalize(c) for c in corpus_texts]
        # if self.lower_case:
        #     corpus_texts = [c.lower() for c in corpus_texts]

        allemb = []
        nbatch = (len(corpus_texts) - 1) // batch_size + 1
        with torch.no_grad():
            # Single card environment or main process display progress bar
            if not dist.is_initialized() or dist.get_rank() == 0:
                iterator = tqdm(range(nbatch), desc="Encoding Corpus")
            else:
                iterator = range(nbatch)
            
            for k in iterator:
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(corpus_texts))
                sub_texts = [self.doc_prefix + text for text in corpus_texts[start_idx:end_idx]]
                sub_texts = [text if text.strip() else "[PAD]" for text in sub_texts] # Some corpus_texts are empty strings, which will cause Diver to report an error

                emb = self.encoder.encode(sub_texts )
                emb = to_numpy(emb)
                allemb.append(emb)

                # cencode = self.tokenize_text(sub_texts, max_length=self.p_max_len, add_special_tokens=self.add_special_tokens, return_tensors="pt")
                # cencode = {key: value.cuda() for key, value in cencode.items()}
                # emb = self.doc_encoder.encode_passage(cencode)
                # allemb.append(emb.cpu())

        allemb = np.vstack(allemb)
        # allemb = torch.cat(allemb, dim=0)
        # allemb = allemb.cuda()
        # if dist.is_initialized():
        #     allemb = dist_utils.varsize_gather_nograd(allemb)
        # allemb = allemb.cpu().numpy()
        return allemb


class SentenceEncoderModel_Prompt:
    '''
    The input model must be a SentenceTransformer model, and it must have the encoder.encode function
    '''

    def __init__(self, encoder, prompts: dict[str, str] = None, model_name: str = None, ):
        self.encoder = encoder
        self.sep = " "
        if prompts:
            self.query_prefix = prompts.get("query", "")
            self.doc_prefix = prompts.get("passage", "")
        else:
            self.query_prefix = ""
            self.doc_prefix = ""

        self.model_name = model_name

    def model_encode(self, sentence, prefix, convert_to_numpy=False ):
        # I have unified all the calling methods
        return self.encoder.encode(sentence, prompt=prefix, convert_to_numpy=convert_to_numpy)  # they use prompt

        # if self.model_name in ["bge_m3", "reasonir", "diver",'qwen3','linq', 'gte','bge_reasoner']:
        #     return self.encoder.encode(sentence, prompt=prefix, convert_to_numpy=convert_to_numpy) # they use prompt
        # elif self.model_name in ["contriever",]:
        #     return self.encoder.encode(sentence, convert_to_numpy=convert_to_numpy,show_progress_bar=False)


    def encode_queries(self, queries: List[str], batch_size: Optional[int] = None, **kwargs) -> np.ndarray:

        # Explicit parameters take precedence, then get from kwargs, then give default values
        if batch_size is None:
            batch_size = kwargs.pop("batch_size", 16)  # Can be replaced with self.default_batch_size

        if dist.is_initialized():
            idx = np.array_split(range(len(queries)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(queries))

        queries = [queries[i] for i in idx]

        allemb = []
        nbatch = (len(queries) - 1) // batch_size + 1
        with torch.no_grad():
            # Single card environment or main process display progress bar
            if not dist.is_initialized() or dist.get_rank() == 0:
                iterator = tqdm(range(nbatch), desc="Encoding Queries")
            else:
                iterator = range(nbatch)

            for k in iterator:
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(queries))
                sub_texts = [text if text.strip() else "[PAD]" for text in queries[start_idx:end_idx]]
                emb = self.model_encode(sub_texts, self.query_prefix)
                emb = to_numpy(emb)  # Convert all to numpy
                allemb.append(emb)

        allemb = np.vstack(allemb)
        # allemb = torch.cat(allemb, dim=0)
        # allemb = allemb.cuda()
        # if dist.is_initialized():
        #     allemb = dist_utils.varsize_gather_nograd(allemb)
        # allemb = allemb.cpu().numpy()
        return allemb

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: Optional[int] = None, **kwargs):
        # Explicit parameters take precedence, then get from kwargs, then give default values
        if batch_size is None:
            batch_size = kwargs.pop("batch_size", 128)  # Can be replaced with self.default_batch_size

        if dist.is_initialized():
            idx = np.array_split(range(len(corpus)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(corpus))
        corpus = [corpus[i] for i in idx]  # Each element is a dictionary, containing "title" and "text"
        corpus_texts = [c["title"] + self.sep + c["text"] if len(c["title"]) > 0 else c["text"] for c in corpus]

        allemb = []
        nbatch = (len(corpus_texts) - 1) // batch_size + 1
        with torch.no_grad():
            # Single card environment or main process display progress bar
            if not dist.is_initialized() or dist.get_rank() == 0:
                iterator = tqdm(range(nbatch), desc="Encoding Corpus")
            else:
                iterator = range(nbatch)

            for k in iterator:
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(corpus_texts))
                sub_texts = [text if text.strip() else "[PAD]" for text in corpus_texts[start_idx:end_idx]]
                emb = self.model_encode(sub_texts, self.doc_prefix)
                emb = to_numpy(emb)
                allemb.append(emb)

        allemb = np.vstack(allemb)
        # allemb = torch.cat(allemb, dim=0)
        # allemb = allemb.cuda()
        # if dist.is_initialized():
        #     allemb = dist_utils.varsize_gather_nograd(allemb)
        # allemb = allemb.cpu().numpy()
        return allemb
