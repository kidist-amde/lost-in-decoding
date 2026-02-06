"""
RQ3 Trained Extension: Query-side planner alignment under cross-lingual query shift.

EXTENSION (not reproduction). Evaluation protocol matches inference-only baselines.

Goal:
  Train a student model so that a non-English query q_lang produces planner-token
  weights aligned to the teacher model's weights for the paired English query q_en
  (same qid), without any re-indexing.

Frozen artifacts (unchanged):
  - Released sequential DocIDs, trie (L=8, V=2048), docid mappings
  - Released docid_to_tokenids.json (top-64 tokens per docid)
  - All index/identifier artifacts and doc-side cached planner artifacts

Teacher = released English PAG checkpoint, frozen.
Student = same initialization, fine-tune ONLY query-side parameters.
"""
