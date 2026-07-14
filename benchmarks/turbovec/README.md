# turbovec benchmark harness

Measures turbovec (Google TurboQuant algorithm, third-party Rust lib by Ryan Codrai)
against exact float32 search **on the real vault embeddings**. Built 2026-06-06 to
answer "should the vault move to a quantized index?" — re-run whenever the vault
grows an order of magnitude.

## Verdict at 8,500 vectors (2026-06-06, turbovec 0.7.0)

| | exact (LanceDB-equivalent) | turbovec 4-bit | turbovec 2-bit |
|---|---|---|---|
| recall@10 | 1.000 | 0.949 | 0.875 |
| p50 latency/query | 0.203 ms | 0.421 ms | 0.228 ms |
| memory | 26.1 MB | 3.3 MB | 1.67 MB |

**WAIT.** At vault scale, quantization trades recall (−5 to −12.5 pts) for memory
nobody needs and adds no speed (exact scan is already 0.2 ms). Becomes interesting
at ~1M+ vectors — i.e. the life-analytics / dense video-embedding pipeline
(~31M vectors/yr at 1 emb/sec), not the text vault. 768-d (nomic-embed-text) is a
friendly dimension for TurboQuant's math.

Full research report: vault `inbox/turboquant-turbovec-research-2026-06-06.md`
(arXiv 2504.19874; lib provenance: third-party alpha, NOT Google).

## Re-run

```bash
uv venv /tmp/tvbench/.venv
uv pip install --python /tmp/tvbench/.venv/bin/python turbovec numpy
~/alfred-v2/.venv/bin/python export.py      # read-only export of vault vectors
/tmp/tvbench/.venv/bin/python bench.py      # prints JSON results
```

Notes:
- `export.py` reads LanceDB table `vault_v2` (current version only), writes
  `/tmp/tvbench/vecs.npy`.
- `bench.py` optionally picks up `/tmp/tvbench/qtext.npy` (real text queries
  embedded via Ollama `nomic-embed-text`, no task prefix — mirrors
  `alfred.embed.ollama`) for a second recall column.
- turbovec `search()` requires a 2-D C-contiguous float32 batch.
- Ground truth = exact cosine top-10 (L2-normalized inner product), 500 held-out
  vault vectors, self-hit excluded.
