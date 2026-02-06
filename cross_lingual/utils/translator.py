"""
Query translation utilities for RQ3.

Provides cached translation of non-English queries to English using either:
1. NLLB-200 (No Language Left Behind) - Facebook's multilingual model
2. M2M100 - Facebook's many-to-many translation model

Translations are cached to disk to avoid redundant computation.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "translation_cache"

# Language code mappings for different translation models
# NLLB uses longer codes (e.g., 'fra_Latn'), M2M100 uses ISO codes
NLLB_LANG_CODES = {
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "zh": "zho_Hans",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "ru": "rus_Cyrl",
    "ja": "jpn_Jpan",
    "ar": "arb_Arab",
    "en": "eng_Latn",
}

M2M100_LANG_CODES = {
    "fr": "fr",
    "de": "de",
    "zh": "zh",
    "es": "es",
    "it": "it",
    "pt": "pt",
    "nl": "nl",
    "ru": "ru",
    "ja": "ja",
    "ar": "ar",
    "en": "en",
}


# ---------------------------------------------------------------------------
# Translation Models
# ---------------------------------------------------------------------------

class BaseTranslator:
    """Base class for translation models."""

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str = "en",
    ) -> str:
        """Translate a single text."""
        raise NotImplementedError

    def translate_batch(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str = "en",
        batch_size: int = 32,
    ) -> List[str]:
        """Translate a batch of texts."""
        raise NotImplementedError


class NLLBTranslator(BaseTranslator):
    """NLLB-200 translator (facebook/nllb-200-distilled-600M)."""

    MODEL_NAME = "facebook/nllb-200-distilled-600M"

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._load_model()

    def _load_model(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        print(f"[NLLBTranslator] Loading {self.MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        print(f"[NLLBTranslator] Model loaded on {self.device}")

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str = "en",
    ) -> str:
        src_code = NLLB_LANG_CODES.get(src_lang, src_lang)
        tgt_code = NLLB_LANG_CODES.get(tgt_lang, tgt_lang)

        self.tokenizer.src_lang = src_code
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_code],
                max_new_tokens=256,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def translate_batch(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str = "en",
        batch_size: int = 32,
    ) -> List[str]:
        src_code = NLLB_LANG_CODES.get(src_lang, src_lang)
        tgt_code = NLLB_LANG_CODES.get(tgt_lang, tgt_lang)

        self.tokenizer.src_lang = src_code
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_code],
                    max_new_tokens=256,
                )

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(decoded)

        return results


class M2M100Translator(BaseTranslator):
    """M2M100 translator (facebook/m2m100_418M)."""

    MODEL_NAME = "facebook/m2m100_418M"

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._load_model()

    def _load_model(self):
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

        print(f"[M2M100Translator] Loading {self.MODEL_NAME}...")
        self.tokenizer = M2M100Tokenizer.from_pretrained(self.MODEL_NAME)
        self.model = M2M100ForConditionalGeneration.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        print(f"[M2M100Translator] Model loaded on {self.device}")

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str = "en",
    ) -> str:
        src_code = M2M100_LANG_CODES.get(src_lang, src_lang)
        tgt_code = M2M100_LANG_CODES.get(tgt_lang, tgt_lang)

        self.tokenizer.src_lang = src_code
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.get_lang_id(tgt_code),
                max_new_tokens=256,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def translate_batch(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str = "en",
        batch_size: int = 32,
    ) -> List[str]:
        src_code = M2M100_LANG_CODES.get(src_lang, src_lang)
        tgt_code = M2M100_LANG_CODES.get(tgt_lang, tgt_lang)

        self.tokenizer.src_lang = src_code
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.get_lang_id(tgt_code),
                    max_new_tokens=256,
                )

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(decoded)

        return results


# ---------------------------------------------------------------------------
# Cached Translation
# ---------------------------------------------------------------------------

def get_cache_key(text: str, src_lang: str, tgt_lang: str, model: str) -> str:
    """Generate a unique cache key for a translation."""
    content = f"{model}:{src_lang}:{tgt_lang}:{text}"
    return hashlib.md5(content.encode()).hexdigest()


def get_cache_path(src_lang: str, tgt_lang: str, model: str) -> Path:
    """Get the cache file path for a language pair and model."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{model}_{src_lang}_to_{tgt_lang}.json"


def load_translation_cache(
    src_lang: str,
    tgt_lang: str = "en",
    model: str = "nllb",
) -> Dict[str, str]:
    """Load cached translations from disk."""
    cache_path = get_cache_path(src_lang, tgt_lang, model)
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_translation_cache(
    cache: Dict[str, str],
    src_lang: str,
    tgt_lang: str = "en",
    model: str = "nllb",
) -> None:
    """Save translations to disk cache."""
    cache_path = get_cache_path(src_lang, tgt_lang, model)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


class CachedTranslator:
    """Translator with disk caching for deterministic, efficient translation."""

    def __init__(
        self,
        model_type: str = "nllb",
        device: str = "cuda",
        cache_only: bool = False,
    ):
        """
        Args:
            model_type: 'nllb' or 'm2m100'
            device: 'cuda' or 'cpu'
            cache_only: If True, only use cached translations (no model loading)
        """
        self.model_type = model_type
        self.cache_only = cache_only
        self._translator: Optional[BaseTranslator] = None
        self.device = device

    def _get_translator(self) -> BaseTranslator:
        """Lazy-load the translation model."""
        if self._translator is None:
            if self.cache_only:
                raise RuntimeError(
                    "Translator is in cache_only mode but translation required. "
                    "Run translate_queries() first to populate cache."
                )
            if self.model_type == "nllb":
                self._translator = NLLBTranslator(self.device)
            elif self.model_type == "m2m100":
                self._translator = M2M100Translator(self.device)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
        return self._translator

    def translate_queries(
        self,
        queries: Dict[str, str],
        src_lang: str,
        tgt_lang: str = "en",
        batch_size: int = 32,
        use_cache: bool = True,
    ) -> Dict[str, str]:
        """Translate a dict of queries with caching.

        Args:
            queries: Dict mapping qid -> query text in source language
            src_lang: Source language code
            tgt_lang: Target language code (default: 'en')
            batch_size: Batch size for translation
            use_cache: Whether to use/update disk cache

        Returns:
            Dict mapping qid -> translated query text
        """
        # Load existing cache
        cache = load_translation_cache(src_lang, tgt_lang, self.model_type) if use_cache else {}

        # Separate cached and uncached queries
        results = {}
        to_translate = {}

        for qid, text in queries.items():
            if text in cache:
                results[qid] = cache[text]
            else:
                to_translate[qid] = text

        if to_translate:
            print(f"[CachedTranslator] {len(results)} cached, "
                  f"{len(to_translate)} to translate ({src_lang} -> {tgt_lang})")

            # Translate uncached queries
            translator = self._get_translator()
            qids = list(to_translate.keys())
            texts = list(to_translate.values())

            translations = translator.translate_batch(
                texts, src_lang, tgt_lang, batch_size
            )

            # Update results and cache
            for qid, orig_text, trans_text in zip(qids, texts, translations):
                results[qid] = trans_text
                cache[orig_text] = trans_text

            # Save updated cache
            if use_cache:
                save_translation_cache(cache, src_lang, tgt_lang, self.model_type)
                print(f"[CachedTranslator] Cache updated: {len(cache)} entries")
        else:
            print(f"[CachedTranslator] All {len(results)} queries found in cache")

        return results


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def translate_mmarco_queries_to_english(
    queries: Dict[str, str],
    src_lang: str,
    model_type: str = "nllb",
    batch_size: int = 32,
    device: str = "cuda",
) -> Dict[str, str]:
    """Translate mMARCO queries back to English.

    This is used for the "translate-at-inference" baseline (System C).

    Args:
        queries: Dict mapping qid -> query text in source language
        src_lang: Source language code (e.g., 'fr', 'de', 'zh')
        model_type: 'nllb' or 'm2m100'
        batch_size: Batch size for translation
        device: Device to use

    Returns:
        Dict mapping qid -> English translation
    """
    translator = CachedTranslator(model_type=model_type, device=device)
    return translator.translate_queries(
        queries=queries,
        src_lang=src_lang,
        tgt_lang="en",
        batch_size=batch_size,
    )


def prepare_translated_queries_for_pag(
    language: str,
    split: str,
    output_base: Path,
    model_type: str = "nllb",
    batch_size: int = 32,
    device: str = "cuda",
) -> Tuple[Path, int]:
    """Prepare back-translated query files for PAG pipeline.

    Translates mMARCO queries from target language back to English
    and saves them in PAG-compatible format.

    Args:
        language: Source language code
        split: Evaluation split
        output_base: Base directory for output
        model_type: Translation model to use
        batch_size: Batch size for translation
        device: Device to use

    Returns:
        (translated_query_dir, n_queries)
    """
    from cross_lingual.data.mmarco_loader import load_mmarco_queries

    # Load mMARCO queries in target language
    queries = load_mmarco_queries(language, split=split)

    # Translate to English
    translated = translate_mmarco_queries_to_english(
        queries=queries,
        src_lang=language,
        model_type=model_type,
        batch_size=batch_size,
        device=device,
    )

    # Write to PAG format
    split_label = {
        "dl19": "TREC_DL_2019",
        "dl20": "TREC_DL_2020",
        "dev": "msmarco_dev",
    }[split]

    translated_dir = output_base / split_label / f"{language}_translated_{model_type}"
    translated_dir.mkdir(parents=True, exist_ok=True)

    translated_tsv = translated_dir / "raw.tsv"
    with open(translated_tsv, "w") as f:
        for qid in sorted(translated.keys(), key=int):
            text = translated[qid].replace("\t", " ").replace("\n", " ")
            f.write(f"{qid}\t{text}\n")

    print(f"[translator] Saved {len(translated)} translated queries to {translated_dir}")
    return translated_dir, len(translated)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Translate mMARCO queries to English")
    parser.add_argument("--language", required=True, help="Source language (fr, de, zh)")
    parser.add_argument("--split", default="dl19", help="Evaluation split")
    parser.add_argument("--model", default="nllb", choices=["nllb", "m2m100"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    output_base = Path(args.output_dir) if args.output_dir else REPO_ROOT / "experiments" / "RQ3_crosslingual" / "queries"

    from cross_lingual.data.mmarco_loader import load_mmarco_queries

    queries = load_mmarco_queries(args.language, split=args.split)
    print(f"Loaded {len(queries)} {args.language} queries for {args.split}")

    translated = translate_mmarco_queries_to_english(
        queries=queries,
        src_lang=args.language,
        model_type=args.model,
        batch_size=args.batch_size,
    )

    print(f"\nSample translations ({args.language} -> en):")
    for qid in list(queries.keys())[:3]:
        print(f"  {qid}:")
        print(f"    {args.language}: {queries[qid]}")
        print(f"    en: {translated[qid]}")
