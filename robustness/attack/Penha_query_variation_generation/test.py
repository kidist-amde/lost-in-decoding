from typing import Dict, List, Optional, Sequence, Tuple
import gc
import torch

from transformations_synonym import SynonymActions
from transformations_paraphrase import ParaphraseActions
from transformations_naturality import NaturalityActions
from transformations_mispelling import MispellingActions
from transformations_ordering import OrderingActions

DEFAULT_QUERIES = ["What is the capital of France?", "Which country you want to visit?"]
# Update this path if you have a fine-tuned paraphraser checkpoint on disk.
DEFAULT_UQV_MODEL_PATH = "./"

QueryVariant = List[Tuple[int, str, str, str, str]]


def _prepare_inputs(
    queries: Optional[Sequence[str]] = None, q_ids: Optional[Sequence[int]] = None
) -> Tuple[List[str], List[int]]:
    """Ensure the helper functions always receive aligned queries/q_ids."""
    queries = list(queries) if queries is not None else list(DEFAULT_QUERIES)
    if not queries:
        raise ValueError("At least one query is required to run the samples.")
    if q_ids is None:
        q_ids = list(range(1, len(queries) + 1))
    else:
        q_ids = list(q_ids)
    if len(queries) != len(q_ids):
        raise ValueError(
            f"queries ({len(queries)}) and q_ids ({len(q_ids)}) must have the same length."
        )
    return queries, q_ids


def test_synonym(queries: Optional[Sequence[str]] = None, q_ids: Optional[Sequence[int]] = None):
    queries, q_ids = _prepare_inputs(queries, q_ids)
    synonym_actions = SynonymActions(queries, q_ids)
    return synonym_actions.adversarial_synonym_replacement() #no problem


def test_paraphrase(
    queries: Optional[Sequence[str]] = None,
    q_ids: Optional[Sequence[int]] = None,
    uqv_model_path: str = DEFAULT_UQV_MODEL_PATH,
):
    queries, q_ids = _prepare_inputs(queries, q_ids)
    paraphrase_actions = ParaphraseActions(queries, q_ids, uqv_model_path=uqv_model_path)

    paraphrase_result_1= paraphrase_actions.back_translation_paraphrase()
    paraphrase_result_2= paraphrase_actions.seq2seq_paraphrase()
    return paraphrase_result_1 + paraphrase_result_2 # no problem


def test_naturality(queries: Optional[Sequence[str]] = None, q_ids: Optional[Sequence[int]] = None):
    queries, q_ids = _prepare_inputs(queries, q_ids)
    naturality_actions = NaturalityActions(queries, q_ids)

    naturality_result_1= naturality_actions.remove_stop_words() # no problem
    # naturality_result_2= naturality_actions.naturality_by_trec_desc_to_title() # the model is not available now
    return naturality_result_1


def test_mispelling(queries: Optional[Sequence[str]] = None, q_ids: Optional[Sequence[int]] = None):
    queries, q_ids = _prepare_inputs(queries, q_ids)
    mispelling_actions = MispellingActions(queries, q_ids)
    return mispelling_actions.mispelling_chars() # no problem


def test_ordering(queries: Optional[Sequence[str]] = None, q_ids: Optional[Sequence[int]] = None):
    queries, q_ids = _prepare_inputs(queries, q_ids)
    ordering_actions = OrderingActions(queries, q_ids)
    return ordering_actions.shuffle_word_order() # no problem


def test_all(
    queries: Optional[Sequence[str]] = None,
    q_ids: Optional[Sequence[int]] = None,
    uqv_model_path: str = DEFAULT_UQV_MODEL_PATH,
) -> Dict[str, QueryVariant]:
    """Run every transformation and return the generated variations."""
    queries, q_ids = _prepare_inputs(queries, q_ids)
    return {
        "paraphrase": test_paraphrase(queries, q_ids, uqv_model_path=uqv_model_path), # must be tested first test_paraphrase, otherwise the memory will be insufficient
        "synonym": test_synonym(queries, q_ids),
        "naturality": test_naturality(queries, q_ids),
        "mispelling": test_mispelling(queries, q_ids),
        "ordering": test_ordering(queries, q_ids),
    }


def print_samples(
    variations: Dict[str, QueryVariant], max_examples: int = 6
) -> None:
    """Pretty print the first few variations from each transformation."""
    for transform, rows in variations.items():
        print(f"\n[{transform}] total variations: {len(rows)}")
        for q_id, original, rewritten, method, category in rows[:max_examples]:
            print(
                f"- q_id={q_id} | {method} ({category})\n"
                f"  original : {original}\n"
                f"  variation: {rewritten}"
            )


if __name__ == "__main__":
    samples = test_all()
    print_samples(samples)
