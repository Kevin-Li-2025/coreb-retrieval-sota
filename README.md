# CoREB Retrieval SOTA

Reproducible CoREB `v202603` public-snapshot retrieval SOTA with C2LLM-7B dense
retrieval under the official query-weighted `nDCG@10` protocol.

## Result

Clean frozen test/public-snapshot result:

| Split | Overall nDCG@10 | Recall@10 | MAP@10 |
| --- | ---: | ---: | ---: |
| `release_v2603` | `0.633174` | `0.821668` | `0.561064` |

Per-task `nDCG@10`:

| Task | nDCG@10 |
| --- | ---: |
| `text2code` | `0.444714` |
| `code2code` | `0.657871` |
| `code2text` | `0.803820` |

The CoREB project page reports the best `v202603` public-snapshot retrieval
overall as `0.624` for GemEmb-2. This run obtains `0.633174` with an open
C2LLM-7B dense retriever.

## Protocol

- Dataset: `hq-bench/coreb`
- Tuning split: `release_v2602`
- Frozen test/public snapshot: `release_v2603`
- Primary metric: query-count-weighted `nDCG@10`
- Code2Code anchor exclusion: enabled, matching the official CoREB runner

For Code2Code, each query has an anchor code item in the shared corpus. The
official runner removes that anchor before computing metrics; this repository
does the same through `TaskData.exclude_doc_ids`.

## Reproduce

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[gpu,dev]"

PYTHONPATH=compat:src python scripts/run_dense_retrieval.py \
  --model /path/to/codefuse-ai_C2LLM-7B \
  --split release_v2603 \
  --tasks all \
  --top-k 128 \
  --eval-k 10 \
  --batch-size 8 \
  --max-length 2048 \
  --dtype bf16 \
  --device cuda \
  --local-files-only \
  --instruction-mode none \
  --output reports/dense_c2llm7b_release_v2603_none_top128_official.json
```

The exact verified report is stored at
`reports/dense_c2llm7b_release_v2603_none_top128_official.json`.

## Validation-Only Follow-Up

The best post-test validation experiment on `release_v2602` reaches
`0.643800` overall `nDCG@10` using dense RRF for `text2code`, Qwen3/dense fusion
for `code2code`, and C2LLM dense retrieval for `code2text`.

This validation number is not used as a clean `release_v2603` test claim.
