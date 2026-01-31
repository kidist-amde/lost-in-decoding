from typing import Dict, Tuple
from tqdm.autonotebook import tqdm
# import json
import ujson as json # Using ujson instead of json, faster
import os
import logging
import csv
from itertools import islice
from torch.utils.data import Dataset, DataLoader
import torch
import mmap
from typing import Union, Iterable,Any      # Add
logger = logging.getLogger(__name__)


class BeirGenericDataLoader: # special for BEAR datasets

    def __init__(self, data_folder: str = "", prefix: str = "", corpus_file: str = "corpus.jsonl",
                 query_file: str = "queries.jsonl",
                 qrels_folder: str = "qrels", qrels_file: str = ""):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}

        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else ''
        self.qrels_file = qrels_file

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError("File {} not present! Please provide accurate file.".format(fIn))

        if not fIn.endswith(ext):
            raise ValueError("File {} must be present with extension {}".format(fIn, ext))

    def load_custom(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d Queries.", len(self.queries))
            logger.info("Query Example: %s", list(self.queries.values())[0])

        return self.corpus, self.queries, self.qrels

    def load(self, split="test") -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        if os.path.exists(self.qrels_file):
            self._load_qrels() # Load qrels
            self.queries = {qid: self.queries[qid] for qid in self.qrels} # 只保留qrels中的query
            logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
            logger.info("Query Example: %s", list(self.queries.values())[0])

        return self.corpus, self.queries, self.qrels

    def load_corpus(self) -> Dict[str, Dict[str, str]]:

        self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        return self.corpus

    # def _load_corpus(self): # This is the original code before optimization
    #     num_lines = sum(1 for i in open(self.corpus_file, 'rb'))
    #     with open(self.corpus_file, encoding='utf8') as fIn:
    #         for line in tqdm(fIn, total=num_lines):
    #             line = json.loads(line)
    #             self.corpus[line.get("_id")] = {
    #                 "text": line.get("text"),
    #                 "title": line.get("title"),
    #             }

    def _load_corpus(self): # This is the optimized code I wrote
        with open(self.corpus_file, encoding='utf8') as fIn:
            lines = fIn.readlines() # Dynamically calculate the number of lines during the file reading process:
            total_lines = len(lines)

            for line in tqdm(lines, total=total_lines):
                line = json.loads(line)
                # If the keys "_id", "text", "title" are certain to exist, the get() method can be avoided, and the dictionary fields can be accessed directly to improve speed.
                self.corpus[line["_id"]] = {
                    "text": line["text"],
                    "title": line["title"],
                }

    def _load_queries(self):

        with open(self.query_file, encoding='utf8') as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[line.get("_id")] = line.get("text")

    def _load_qrels(self):

        open_file = open(self.qrels_file, encoding="utf-8")  # Manually open the file

        reader = csv.reader(open_file,delimiter="\t", quoting=csv.QUOTE_MINIMAL)

        next(reader)  # Skip the header

        for id, row in enumerate(reader):
        # for id, row in enumerate(islice(reader, 10000)): # Only read the first 10000 lines of data
            query_id, corpus_id, score = row[0], row[1], int(row[2])

            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score

        open_file.close() # Manually close the file to ensure resource release

class BeirEncodeDataset(Dataset):
    def __init__(self, text=None,*,corpus=None, query=None, tokenizer, max_length=None):
        # text is the first positional parameter, other parameters are separated by *, ensuring that corpus, query, etc. other parameters can only be passed as keyword parameters to avoid confusion.
        if corpus:
            self.corpus_texts = ['{} {}'.format(corpus[doc].get('title', ''), corpus[doc]['text']).strip() for doc in corpus]
            self.id = [doc for doc in corpus]
            self.encode_passage = True
            self.encode_query = False
        elif query:
            self.corpus_texts = list(query.values())
            self.id = list(query.keys())
            self.encode_passage = False
            self.encode_query = True
        else:
            self.corpus_texts = text
            self.id = None
            self.encode_passage = False
            self.encode_query = False
        self.tokenizer = tokenizer # tokenizer is a special-processed tokenizer, encode_query=False, encode_passage=False need to be set
        self.max_length = max_length

    def __len__(self):
        return len(self.corpus_texts) if self.corpus_texts is not None else 0

    def __getitem__(self, idx):
        # Get the text at the specified index and use the tokenizer to tokenize
        if self.corpus_texts is None:
            raise ValueError("corpus_texts is None. Please provide valid text, corpus, or query bright_data.")
        text = self.corpus_texts[idx]
        tokenized_text = self.tokenizer(text, encode_query=self.encode_query, encode_passage=self.encode_passage, padding="max_length", truncation=True, max_length=self.max_length,
                                        return_tensors="pt")
        # Return a dictionary containing input_ids and attention_mask
        return {k: v.squeeze(0) for k, v in tokenized_text.items()}

class Attack_Batch_Dataset(Dataset):
    def __init__(self, data):
        # 将 numpy 数组转换为 torch 张量
        self.data = torch.from_numpy(data).float().to('cuda')

    def __len__(self):
        # 返回数据集的大小
        return len(self.data)

    def __getitem__(self, index):
        # 返回对应 index 的数据
        return self.data[index]


# class DiffuseIRDataset(Dataset):
#     def __init__(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], tokenizer, max_length=512):
#         self.corpus = corpus
#         self.queries = queries
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#
#     def __len__(self):
#         return len(self.queries)
#
#     def __getitem__(self, idx):
#         query_id = list(self.queries.keys())[idx]
#         query_text = self.queries[query_id]


class GenericDataLoader:
    '''
    I wrote it myself, originated from from beir.datasets.data_loader_yk import GenericDataLoader
    The main difference between the following and BeirGenericDataLoader is that the following supports outputting multiple splits
    '''
    def __init__(
        self,
        data_folder: str = "",
        prefix: str = "",
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        qrels_folder: str = "qrels",
        qrels_file: str = "",
    ):
        self.corpus = {}
        self.queries = {}
        self.queries_raw=[] # For bright datasets
        self.qrels = {} # Originally {}, now a list containing dicts

        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else ""
        self.qrels_file = qrels_file

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError(f"File {fIn} not present! Please provide accurate file.")

        if not fIn.endswith(ext):
            raise ValueError(f"File {fIn} must be present with extension {ext}")

    # def load_custom(
    #     self,
    # ) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
    #     self.check(fIn=self.corpus_file, ext="jsonl")
    #     self.check(fIn=self.query_file, ext="jsonl")
    #     self.check(fIn=self.qrels_file, ext="tsv")
    #
    #     if not len(self.corpus):
    #         logger.info("Loading Corpus...")
    #         self._load_corpus()
    #         logger.info("Loaded %d Documents.", len(self.corpus))
    #         logger.info("Doc Example: %s", list(self.corpus.values())[0])
    #
    #     if not len(self.queries):
    #         logger.info("Loading Queries...")
    #         self._load_queries()
    #
    #     if os.path.exists(self.qrels_file):
    #         self._load_qrels()
    #         self.queries = {qid: self.queries[qid] for qid in self.qrels}
    #         logger.info("Loaded %d Queries.", len(self.queries))
    #         logger.info("Query Example: %s", list(self.queries.values())[0])
    #
    #     return self.corpus, self.queries, self.qrels

    def load(self, split: str | list[str] = "test" ) -> Dict[str, Any]:

        # 2. Unify to a list, for easier subsequent traversal
        if isinstance(split, str):
            splits: Iterable[str] = [split]
        else:
            splits = split

        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")


        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        self.queries_split = {}
        for sp_name in splits:
            qrels_file= os.path.join(self.qrels_folder, f"{sp_name}.tsv")
            if not  (os.path.exists(qrels_file)):
                raise ValueError(f"File {qrels_file} not present! Please provide accurate file.")
            # Load qrels directly because qrels_file already exists
            qrels = self._load_qrels(qrels_file)
            self.queries_split.update( {qid: self.queries[qid] for qid in  qrels})
            self.qrels[sp_name]= qrels

        logger.info("Loaded %d %s Queries.", len(self.queries_split), split )
        logger.info("Query Example: %s", list(self.queries_split.values())[0])

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split )
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        if len(self.qrels)==1:
            output={
                'corpus': self.corpus,
                'queries': self.queries_split,
                'qrels': self.qrels[splits[0]],
                'queries_raw': self.queries_raw,
            }
        else:
            output={
                'corpus': self.corpus,
                'queries': self.queries_split,
                'qrels': self.qrels,
                'queries_raw': self.queries_raw,
            }
        return output

    def load_corpus(self) -> dict[str, dict[str, str]]:
        self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        return self.corpus

    def _load_corpus(self):
        num_lines = sum(1 for i in open(self.corpus_file, "rb"))
        with open(self.corpus_file, encoding="utf8") as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                self.corpus[line.get("_id")] = {
                    "text": line.get("text"),
                    "title": line.get("title"),
                }

    def _load_queries(self):
        with open(self.query_file, encoding="utf8") as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[line.get("_id")] = line.get("text")
                self.queries_raw.append(line) # For bright datasets

    def _load_qrels(self, qrels_file: str = ""):
        reader = csv.reader(
            open( qrels_file, encoding="utf-8"),
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
        )
        next(reader)

        qrels= {}
        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])

            if query_id not in  qrels:
                qrels[query_id] = {corpus_id: score}
            else:
                qrels[query_id][corpus_id] = score
        return qrels
