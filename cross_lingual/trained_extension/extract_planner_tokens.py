#!/usr/bin/env python3
"""
Planner token extraction for trained extension.

Reuses the EXACT extraction logic from RQ2 (robustness/utils/pag_inference.py)
to maintain consistency with TokJaccard@100 measurements.

Two modes:
1. Subprocess extraction (via _EXTRACT_TOKENS_SCRIPT) - file-based, same as RQ2
2. In-process extraction with full vocab scores - needed for KL distillation training

Data flow (same as RQ2):
  Query text -> T5 encoder -> T5 decoder (mode="lex_retrieval")
  -> lm_head(sequence_output) -> lexical_logits [bz, seq_len, vocab_size]
  -> log(1 + relu(logits)) * attention_mask
  -> max-pool over seq dim -> lexical_rep [bz, vocab_size]
  -> topk(..., k=100) -> (token_ids, scores)
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from robustness.utils.pag_inference import (
    PRETRAINED_PATH,
    extract_planner_tokens as _subprocess_extract,
    load_planner_tokens,
    load_planner_tokens_with_scores,
)


# ---------------------------------------------------------------------------
# Subprocess extraction (reuses RQ2 exactly)
# ---------------------------------------------------------------------------

def extract_planner_tokens_file(
    query_dir: str,
    out_path: str,
    pretrained_path: str = PRETRAINED_PATH,
    topk: int = 100,
    batch_size: int = 8,
    max_length: int = 128,
) -> Dict[str, Dict]:
    """Extract planner tokens via subprocess (identical to RQ2).

    Returns {qid: {"token_ids": [...], "scores": [...]}}.
    """
    return _subprocess_extract(
        query_dir=query_dir,
        out_path=out_path,
        pretrained_path=pretrained_path,
        topk=topk,
        batch_size=batch_size,
        max_length=max_length,
    )


# ---------------------------------------------------------------------------
# In-process extraction returning full vocab scores
# ---------------------------------------------------------------------------

def load_model_for_extraction(
    pretrained_path: str = PRETRAINED_PATH,
    device: str = "cuda:0",
):
    """Load LexicalRipor model in lex_retrieval mode.

    Returns (model, tokenizer_name).
    """
    from t5_pretrainer.modeling.t5_generative_retriever import LexicalRipor

    model = LexicalRipor.from_pretrained(pretrained_path)
    model.eval()
    model.to(device)
    model.base_model.mode = "lex_retrieval"

    return model, pretrained_path


def extract_full_vocab_scores(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
) -> torch.Tensor:
    """Extract full vocab-size planner scores for a batch.

    This is the same computation as LexicalRipor.encode(), but returns
    the full [bz, vocab_size] tensor (not just top-K).

    Args:
        model: LexicalRipor in eval mode on GPU
        input_ids: [bz, seq_len]
        attention_mask: [bz, seq_len]
        decoder_input_ids: [bz, seq_len] (copy of input_ids for lex_retrieval)

    Returns:
        lexical_rep: [bz, vocab_size] - SPLADE-style scores
    """
    with torch.no_grad():
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
        }
        lexical_rep, _ = model.encode(**inputs)
    return lexical_rep


def extract_full_vocab_scores_grad(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
) -> torch.Tensor:
    """Same as extract_full_vocab_scores but WITH gradients (for student).

    Returns:
        lexical_rep: [bz, vocab_size] - SPLADE-style scores, differentiable
    """
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
    }
    model_output = model.base_model(
        **inputs, return_dict=True,
    )
    lexical_rep, _ = torch.max(
        torch.log(1 + torch.relu(model_output.lexical_logit))
        * attention_mask.unsqueeze(-1),
        dim=1,
    )
    return lexical_rep


# ---------------------------------------------------------------------------
# Batch extraction with DataLoader
# ---------------------------------------------------------------------------

def extract_tokens_from_tsv(
    query_dir: str,
    pretrained_path: str = PRETRAINED_PATH,
    topk: int = 100,
    batch_size: int = 8,
    max_length: int = 128,
    device: str = "cuda:0",
    return_full_scores: bool = False,
) -> Dict[str, Dict]:
    """In-process extraction from a raw.tsv directory.

    Args:
        query_dir: directory containing raw.tsv
        pretrained_path: model checkpoint path
        topk: number of top tokens to return
        batch_size: batch size
        max_length: max token length
        device: GPU device
        return_full_scores: if True, also return full vocab scores

    Returns:
        {qid: {"token_ids": [...], "scores": [...],
                "full_scores": tensor (only if return_full_scores)}}
    """
    from tqdm import tqdm
    from t5_pretrainer.dataset.dataset import CollectionDatasetPreLoad
    from t5_pretrainer.dataset.dataloader import T5SpladeCollectionDataLoader

    model, _ = load_model_for_extraction(pretrained_path, device)

    q_collection = CollectionDatasetPreLoad(data_dir=query_dir, id_style="row_id")
    q_loader = T5SpladeCollectionDataLoader(
        dataset=q_collection, tokenizer_type=pretrained_path,
        max_length=max_length, batch_size=batch_size,
        shuffle=False, num_workers=1,
    )

    results = {}
    for batch in tqdm(q_loader, desc="extract planner tokens"):
        inputs = {k: v.to(device) for k, v in batch.items() if k != "id"}
        lexical_rep = extract_full_vocab_scores(model, **inputs)
        top_scores, top_ids = torch.topk(lexical_rep, k=topk, dim=-1)

        if isinstance(batch["id"], torch.LongTensor):
            query_ids = batch["id"].tolist()
        elif isinstance(batch["id"], list):
            query_ids = batch["id"]
        else:
            query_ids = list(batch["id"])

        for i, qid in enumerate(query_ids):
            entry = {
                "token_ids": top_ids[i].cpu().tolist(),
                "scores": top_scores[i].cpu().tolist(),
            }
            if return_full_scores:
                entry["full_scores"] = lexical_rep[i].cpu()
            results[str(qid)] = entry

    return results


# ---------------------------------------------------------------------------
# CLI: extract teacher/student planner tokens
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract planner tokens (reuses RQ2 extraction)"
    )
    parser.add_argument("--query_dir", type=str, required=True,
                        help="Directory containing raw.tsv")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Output JSON path")
    parser.add_argument("--pretrained_path", type=str, default=PRETRAINED_PATH)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--mode", choices=["subprocess", "inprocess"],
                        default="subprocess",
                        help="subprocess = exact RQ2; inprocess = same logic in-process")
    args = parser.parse_args()

    if args.mode == "subprocess":
        tokens = extract_planner_tokens_file(
            query_dir=args.query_dir,
            out_path=args.out_path,
            pretrained_path=args.pretrained_path,
            topk=args.topk,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
    else:
        tokens = extract_tokens_from_tsv(
            query_dir=args.query_dir,
            pretrained_path=args.pretrained_path,
            topk=args.topk,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
        # Save without full_scores (not serializable)
        save_data = {
            qid: {"token_ids": v["token_ids"], "scores": v["scores"]}
            for qid, v in tokens.items()
        }
        with open(args.out_path, "w") as f:
            json.dump(save_data, f)

    print(f"[extract] Done. {len(tokens)} queries -> {args.out_path}")
