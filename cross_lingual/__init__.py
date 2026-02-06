"""
RQ3: Cross-lingual evaluation of PAG's lexical planner.

Tests whether PAG's lexical planner helps when queries are non-English
but documents remain English (MS MARCO passages).

Languages tested:
- Chinese (zh): hard lexical mismatch
- Dutch (nl): moderate lexical mismatch
- French (fr): moderate lexical mismatch
- German (de): moderate lexical mismatch

Systems evaluated:
A) Naive cross-lingual PAG (no alignment)
B) Sequential-only (planner disabled)
C) Translate-at-inference PAG baseline

Entry points (flat layout):
- data_loader.py         : mMARCO download, qid validation, coverage reports
- retrieval_engine.py    : PAG inference wrapper with latency tracking
- baseline_naive.py      : Baseline A runner
- baseline_sequential.py : Baseline B runner
- baseline_translate.py  : Baseline C runner (with translation log)
- plan_diagnostics.py    : CandOverlap@100, TokJaccard@100
- evaluate.py            : MRR@10, Recall@10, nDCG@10, bootstrap sig. test
- run_rq3_baselines.sh   : Master runner for all baselines x languages
"""
