#!/usr/bin/env python3
"""
Train query-side planner alignment via KL distillation.

Teacher = released English PAG checkpoint (frozen).
Student = same initialization, fine-tune only query-side parameters
          (encoder + lm_head, optionally last N encoder layers only).

Objective:
  For each (q_en, q_lang) pair (same qid):
    T = extract_planner_tokens(q_en)   using frozen teacher
    S = extract_planner_tokens(q_lang)  using student
    L_align = KL(softmax(z_T / tau) || softmax(z_S / tau))
  over token union U = T_ids ∪ S_ids.

Usage:
  python -m cross_lingual.trained_extension.train_planner_alignment \\
      --config cross_lingual/trained_extension/configs/default.yaml

  python -m cross_lingual.trained_extension.train_planner_alignment \\
      --language fr --epochs 3 --lr 1e-5 --temperature 2.0
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from robustness.utils.pag_inference import PRETRAINED_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TOKENIZER_FILES = [
    "spiece.model", "tokenizer.json",
    "special_tokens_map.json", "tokenizer_config.json",
]


def _copy_tokenizer(src_dir: str, dst_dir: str):
    """Copy tokenizer files from source checkpoint to saved checkpoint."""
    for fname in TOKENIZER_FILES:
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fname))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "language": "fr",
    "pretrained_path": str(PRETRAINED_PATH),
    "output_dir": str(REPO_ROOT / "cross_lingual" / "trained_extension" / "checkpoints"),
    "results_dir": str(REPO_ROOT / "cross_lingual" / "trained_extension" / "results"),

    # Training
    "epochs": 5,
    "lr": 1e-5,
    "weight_decay": 0.01,
    "batch_size": 8,
    "grad_accumulation_steps": 4,
    "max_length": 128,
    "warmup_fraction": 0.1,
    "max_grad_norm": 1.0,

    # Distillation
    "temperature": 2.0,
    "topk_union": 100,

    # Freezing
    "finetune_last_n_layers": 12,  # 12 = all encoder layers for t5-base
    "freeze_decoder": True,

    # Logging
    "log_interval": 50,
    "eval_interval": 500,
    "save_interval": 1000,
    "seed": 42,
}


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load config from YAML file, falling back to defaults."""
    config = dict(DEFAULT_CONFIG)
    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path) as f:
            overrides = yaml.safe_load(f) or {}
        config.update(overrides)
        logger.info(f"Loaded config from {config_path}")
    return config


# ---------------------------------------------------------------------------
# Dataset: parallel query pairs with tokenization
# ---------------------------------------------------------------------------

class PlannerAlignmentDataset(Dataset):
    """Dataset yielding tokenized (q_en, q_lang) pairs."""

    def __init__(
        self,
        q_en: Dict[str, str],
        q_lang: Dict[str, str],
        qids: List[str],
        tokenizer,
        max_length: int = 128,
    ):
        self.qids = qids
        self.q_en = q_en
        self.q_lang = q_lang
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, idx):
        qid = self.qids[idx]
        return qid, self.q_en[qid], self.q_lang[qid]


def collate_fn_factory(tokenizer, max_length):
    """Create a collate function for PlannerAlignmentDataset."""
    from copy import deepcopy

    def collate_fn(batch):
        qids, en_texts, lang_texts = zip(*batch)

        # Tokenize English (teacher input)
        en_tok = tokenizer(
            list(en_texts),
            add_special_tokens=True,
            padding="longest",
            truncation="longest_first",
            max_length=max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        # decoder_input_ids = copy of input_ids (for lex_retrieval mode)
        en_tok["decoder_input_ids"] = en_tok["input_ids"].clone()

        # Tokenize target language (student input)
        lang_tok = tokenizer(
            list(lang_texts),
            add_special_tokens=True,
            padding="longest",
            truncation="longest_first",
            max_length=max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        lang_tok["decoder_input_ids"] = lang_tok["input_ids"].clone()

        return {
            "qids": list(qids),
            "en": en_tok,
            "lang": lang_tok,
        }

    return collate_fn


# ---------------------------------------------------------------------------
# Model setup: teacher (frozen) + student (trainable query-side)
# ---------------------------------------------------------------------------

def setup_models(config: Dict, device: str = "cuda:0"):
    """Load teacher and student models.

    Teacher: frozen, eval mode.
    Student: same init, only query-side params trainable.

    Returns (teacher, student, tokenizer)
    """
    from t5_pretrainer.modeling.t5_generative_retriever import LexicalRipor
    from transformers import AutoTokenizer

    pretrained_path = config["pretrained_path"]

    # Teacher (frozen)
    teacher = LexicalRipor.from_pretrained(pretrained_path)
    teacher.eval()
    teacher.to(device)
    teacher.base_model.mode = "lex_retrieval"
    for p in teacher.parameters():
        p.requires_grad = False

    # Student (trainable)
    student = LexicalRipor.from_pretrained(pretrained_path)
    student.to(device)
    student.base_model.mode = "lex_retrieval"

    # Freeze parameters based on config
    _freeze_student_params(student, config)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in student.parameters())
    logger.info(f"Student: {n_trainable:,} / {n_total:,} trainable parameters "
                f"({n_trainable/n_total:.1%})")

    return teacher, student, tokenizer


def _freeze_student_params(student, config: Dict):
    """Freeze decoder and optionally early encoder layers."""
    freeze_decoder = config.get("freeze_decoder", True)
    finetune_last_n = config.get("finetune_last_n_layers", 12)

    # Freeze decoder entirely
    if freeze_decoder:
        for p in student.base_model.decoder.parameters():
            p.requires_grad = False
        logger.info("Frozen: decoder (all layers)")

    # Freeze early encoder layers (keep last N trainable)
    encoder = student.base_model.encoder
    total_layers = len(encoder.block)

    if finetune_last_n < total_layers:
        freeze_up_to = total_layers - finetune_last_n
        # Freeze embedding
        for p in encoder.embed_tokens.parameters():
            p.requires_grad = False
        # Freeze early blocks
        for i in range(freeze_up_to):
            for p in encoder.block[i].parameters():
                p.requires_grad = False
        logger.info(f"Frozen: encoder layers 0-{freeze_up_to-1} "
                    f"(training last {finetune_last_n} of {total_layers})")
    else:
        logger.info(f"Training all {total_layers} encoder layers")

    # lm_head is always trainable (projects decoder hidden states to vocab)
    for p in student.base_model.lm_head.parameters():
        p.requires_grad = True
    logger.info("Trainable: lm_head")


# ---------------------------------------------------------------------------
# KL Distillation Loss
# ---------------------------------------------------------------------------

def compute_kl_alignment_loss(
    teacher_scores: torch.Tensor,
    student_scores: torch.Tensor,
    temperature: float = 2.0,
    topk_union: int = 100,
) -> torch.Tensor:
    """Compute KL distillation loss over union of top-K planner tokens.

    Args:
        teacher_scores: [bz, vocab_size] - teacher planner scores (no grad)
        student_scores: [bz, vocab_size] - student planner scores (with grad)
        temperature: softmax temperature
        topk_union: K for top-K union computation

    Returns:
        scalar loss (mean over batch)
    """
    bz, vocab_size = teacher_scores.shape

    # Get top-K token indices from both teacher and student
    _, teacher_topk_ids = torch.topk(teacher_scores, k=topk_union, dim=-1)  # [bz, K]
    _, student_topk_ids = torch.topk(student_scores.detach(), k=topk_union, dim=-1)  # [bz, K]

    # Build union mask: [bz, vocab_size] with 1s at union positions
    union_mask = torch.zeros(bz, vocab_size, device=teacher_scores.device, dtype=torch.bool)
    union_mask.scatter_(1, teacher_topk_ids, True)
    union_mask.scatter_(1, student_topk_ids, True)

    # For efficiency: gather union scores per sample
    # Find max union size
    union_sizes = union_mask.sum(dim=1)  # [bz]
    max_union = union_sizes.max().item()

    # Build padded union indices
    # Sort union indices for determinism
    VERY_SMALL = -1e9
    losses = []

    for i in range(bz):
        mask_i = union_mask[i]  # [vocab_size]
        indices_i = mask_i.nonzero(as_tuple=True)[0]  # [union_size_i]

        # Gather logits for union tokens
        t_logits = teacher_scores[i, indices_i]  # [union_size_i]
        s_logits = student_scores[i, indices_i]  # [union_size_i]

        # Apply temperature and softmax
        p = F.softmax(t_logits / temperature, dim=0)  # teacher distribution
        log_q = F.log_softmax(s_logits / temperature, dim=0)  # student log distribution

        # KL(p || q) = sum(p * (log_p - log_q))
        kl = F.kl_div(log_q, p, reduction="sum")
        losses.append(kl)

    loss = torch.stack(losses).mean()

    # Scale by T^2 (standard distillation scaling)
    loss = loss * (temperature ** 2)

    return loss


# ---------------------------------------------------------------------------
# Forward pass for planner scores
# ---------------------------------------------------------------------------

def get_teacher_scores(teacher, batch_inputs: Dict) -> torch.Tensor:
    """Get teacher planner scores (frozen, no grad)."""
    with torch.no_grad():
        model_output = teacher.base_model(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            decoder_input_ids=batch_inputs["decoder_input_ids"],
            return_dict=True,
        )
        lexical_rep, _ = torch.max(
            torch.log(1 + torch.relu(model_output.lexical_logit))
            * batch_inputs["attention_mask"].unsqueeze(-1),
            dim=1,
        )
    return lexical_rep  # [bz, vocab_size]


def get_student_scores(student, batch_inputs: Dict) -> torch.Tensor:
    """Get student planner scores (with gradients)."""
    model_output = student.base_model(
        input_ids=batch_inputs["input_ids"],
        attention_mask=batch_inputs["attention_mask"],
        decoder_input_ids=batch_inputs["decoder_input_ids"],
        return_dict=True,
    )
    lexical_rep, _ = torch.max(
        torch.log(1 + torch.relu(model_output.lexical_logit))
        * batch_inputs["attention_mask"].unsqueeze(-1),
        dim=1,
    )
    return lexical_rep  # [bz, vocab_size]


# ---------------------------------------------------------------------------
# Evaluation (TokJaccard + alignment metrics during training)
# ---------------------------------------------------------------------------

def evaluate_alignment(
    teacher,
    student,
    eval_loader: DataLoader,
    device: str,
    topk: int = 100,
) -> Dict[str, float]:
    """Evaluate planner alignment on dev set.

    Returns:
        {
            "tok_jaccard_mean": float,
            "tok_jaccard_median": float,
            "kl_mean": float,
            "n_queries": int,
        }
    """
    student.eval()
    jaccard_vals = []
    kl_vals = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="eval alignment", leave=False):
            en_inputs = {k: v.to(device) for k, v in batch["en"].items()}
            lang_inputs = {k: v.to(device) for k, v in batch["lang"].items()}

            t_scores = get_teacher_scores(teacher, en_inputs)
            s_scores = get_student_scores(student, lang_inputs)

            # TokJaccard@K per query
            _, t_topk = torch.topk(t_scores, k=topk, dim=-1)
            _, s_topk = torch.topk(s_scores, k=topk, dim=-1)

            for i in range(t_scores.size(0)):
                t_set = set(t_topk[i].cpu().tolist())
                s_set = set(s_topk[i].cpu().tolist())
                inter = t_set & s_set
                union = t_set | s_set
                jacc = len(inter) / len(union) if union else 1.0
                jaccard_vals.append(jacc)

            # KL (for monitoring)
            kl = compute_kl_alignment_loss(
                t_scores, s_scores,
                temperature=1.0, topk_union=topk,
            )
            kl_vals.append(kl.item())

    student.train()

    jac_arr = np.array(jaccard_vals)
    return {
        "tok_jaccard_mean": float(np.mean(jac_arr)),
        "tok_jaccard_median": float(np.median(jac_arr)),
        "tok_jaccard_p10": float(np.percentile(jac_arr, 10)),
        "tok_jaccard_p90": float(np.percentile(jac_arr, 90)),
        "kl_mean": float(np.mean(kl_vals)),
        "n_queries": len(jaccard_vals),
    }


# ---------------------------------------------------------------------------
# Learning rate scheduler with warmup
# ---------------------------------------------------------------------------

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup then linear decay."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: Dict):
    """Main training function."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    language = config["language"]
    output_dir = Path(config["output_dir"]) / language
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(config["results_dir"]) / language
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load models
    logger.info("Loading teacher and student models...")
    teacher, student, tokenizer = setup_models(config, device)

    # Load data
    logger.info(f"Loading parallel data for {language}...")
    from cross_lingual.trained_extension.data_loader_parallel import (
        get_train_dev_split,
        ParallelQueryDataset,
        load_mmarco_english,
        load_mmarco_queries,
        validate_alignment,
    )

    train_qids, dev_qids = get_train_dev_split(language, download=True)

    q_en = load_mmarco_english()
    q_lang = load_mmarco_queries(language, download=False)
    aligned = validate_alignment(q_en, q_lang, language)

    # Train set
    train_qid_list = sorted(aligned & train_qids)
    dev_qid_list = sorted(aligned & dev_qids)

    logger.info(f"Train: {len(train_qid_list)} pairs, Dev: {len(dev_qid_list)} pairs")

    train_dataset = PlannerAlignmentDataset(
        q_en, q_lang, train_qid_list, tokenizer, config["max_length"],
    )
    dev_dataset = PlannerAlignmentDataset(
        q_en, q_lang, dev_qid_list, tokenizer, config["max_length"],
    )

    collate_fn = collate_fn_factory(tokenizer, config["max_length"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    # Optimizer
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    total_steps = len(train_loader) * config["epochs"] // config["grad_accumulation_steps"]
    warmup_steps = int(total_steps * config["warmup_fraction"])
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    logger.info(f"Total steps: {total_steps}, warmup: {warmup_steps}")

    # Training loop
    student.train()
    global_step = 0
    best_jaccard = 0.0
    training_log = []

    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, batch in enumerate(tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"
        )):
            en_inputs = {k: v.to(device) for k, v in batch["en"].items()}
            lang_inputs = {k: v.to(device) for k, v in batch["lang"].items()}

            # Teacher scores (frozen, no grad)
            t_scores = get_teacher_scores(teacher, en_inputs)

            # Student scores (with grad)
            s_scores = get_student_scores(student, lang_inputs)

            # KL alignment loss
            loss = compute_kl_alignment_loss(
                t_scores, s_scores,
                temperature=config["temperature"],
                topk_union=config["topk_union"],
            )

            loss = loss / config["grad_accumulation_steps"]
            loss.backward()

            epoch_loss += loss.item() * config["grad_accumulation_steps"]
            epoch_steps += 1

            if (batch_idx + 1) % config["grad_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, config["max_grad_norm"],
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % config["log_interval"] == 0:
                    avg_loss = epoch_loss / epoch_steps
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"Step {global_step} | Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | Epoch: {epoch+1}"
                    )
                    training_log.append({
                        "step": global_step,
                        "loss": avg_loss,
                        "lr": lr,
                        "epoch": epoch + 1,
                    })

                # Evaluation
                if global_step % config["eval_interval"] == 0:
                    eval_metrics = evaluate_alignment(
                        teacher, student, dev_loader, device,
                        topk=config["topk_union"],
                    )
                    logger.info(
                        f"Eval @ step {global_step}: "
                        f"TokJaccard={eval_metrics['tok_jaccard_mean']:.4f} "
                        f"KL={eval_metrics['kl_mean']:.4f}"
                    )
                    eval_metrics["step"] = global_step
                    training_log.append({"eval": eval_metrics})

                    # Save best
                    if eval_metrics["tok_jaccard_mean"] > best_jaccard:
                        best_jaccard = eval_metrics["tok_jaccard_mean"]
                        best_dir = output_dir / "best"
                        student.save_pretrained(str(best_dir))
                        _copy_tokenizer(config["pretrained_path"], str(best_dir))
                        logger.info(
                            f"New best TokJaccard: {best_jaccard:.4f} -> {best_dir}"
                        )

                    student.train()

                # Periodic save
                if global_step % config["save_interval"] == 0:
                    step_dir = output_dir / f"step_{global_step}"
                    student.save_pretrained(str(step_dir))
                    _copy_tokenizer(config["pretrained_path"], str(step_dir))

        # End of epoch eval
        eval_metrics = evaluate_alignment(
            teacher, student, dev_loader, device,
            topk=config["topk_union"],
        )
        logger.info(
            f"Epoch {epoch+1} done: "
            f"TokJaccard={eval_metrics['tok_jaccard_mean']:.4f} "
            f"KL={eval_metrics['kl_mean']:.4f}"
        )
        eval_metrics["epoch"] = epoch + 1
        training_log.append({"epoch_eval": eval_metrics})

        if eval_metrics["tok_jaccard_mean"] > best_jaccard:
            best_jaccard = eval_metrics["tok_jaccard_mean"]
            best_dir = output_dir / "best"
            student.save_pretrained(str(best_dir))
            _copy_tokenizer(config["pretrained_path"], str(best_dir))
            logger.info(f"New best TokJaccard: {best_jaccard:.4f} -> {best_dir}")

        student.train()

    # Save final checkpoint
    final_dir = output_dir / "final"
    student.save_pretrained(str(final_dir))
    _copy_tokenizer(config["pretrained_path"], str(final_dir))
    logger.info(f"Final checkpoint -> {final_dir}")

    # Save training log
    log_path = results_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    logger.info(f"Training log -> {log_path}")

    # Save final summary
    summary = {
        "language": language,
        "best_tok_jaccard": best_jaccard,
        "total_steps": global_step,
        "epochs": config["epochs"],
        "final_eval": eval_metrics,
        "config": config,
    }
    with open(results_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Training complete. Best TokJaccard@100: {best_jaccard:.4f}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train planner alignment via KL distillation"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--topk_union", type=int, default=None)
    parser.add_argument("--finetune_last_n_layers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    for key in ["language", "epochs", "lr", "batch_size", "temperature",
                "topk_union", "finetune_last_n_layers", "output_dir",
                "pretrained_path", "seed"]:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    logger.info(f"Config: {json.dumps(config, indent=2)}")
    train(config)


if __name__ == "__main__":
    main()
