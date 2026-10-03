# Kochi-LLM

A **1.02B-parameter Mixture-of-Experts language model trained entirely from scratch** on 2× Tesla T4
(Kaggle, ~52 GPU-hours), including its own BPE tokenizer, data pipeline, and distributed training setup.

Weights are published on Hugging Face as **KochiLLM-v1.0**.

---

## Results

Trained for exactly one epoch over a 180M-token corpus, with the cosine schedule annealed to completion.

| Metric | Value |
|---|---|
| Total parameters | 1,023M (~634M active per token) |
| Active parameters / token | ~634M (1 shared + top-2 of 4 routed experts) |
| Training tokens | 180,224,000 (0.9996 epochs) |
| Tokens per parameter | 0.176 |
| Throughput | ~37.9 s/step (~865 tok/s, 2× T4) |
| Validation loss | **3.96** (final, annealed) · 3.91 (best) |
| Validation loss at Phase-2 start | 5.64 |
| Phase-2 GPU-hours | ~52 |
| Context length | 1024 tokens |

Validation loss on the FineWeb-Edu / Cosmopedia-v2 holdout:

```
step   800  5.6359
step  1400  4.8471
step  2400  4.7707
step  3000  4.4930
step  3600  4.3739
step  4200  4.2679
step  5200  3.9055   <- best
step  5499  3.9603   <- final, fully annealed (LR = 1e-4)
```

> The ~0.05 gap between best and final is evaluation noise: each eval averages only 10 random
> 1024-token batches. The final checkpoint is preferred because it sits at the schedule endpoint.

## Architecture

| Component | Detail |
|---|---|
| Layers | 12 |
| Model dim | 1536 |
| Attention heads | 24 query / 8 KV (GQA, ratio 3, head_dim 64) |
| FFN | SwiGLU, hidden = 2 × dim |
| Experts | 4 routed (top-2) + 1 shared — DeepSeek-MoE style |
| Normalization | RMSNorm, plus QK-Norm on queries and keys |
| Position | RoPE (θ = 10000) |
| Attention window | alternating: full on even layers, 512-token sliding on odd |
| Soft-capping | attention logits (50) and final logits (30), Gemma-2 style |
| Multi-token prediction | 2 heads (`n_predict=2`), extra head discarded at inference |
| Weight tying | input embedding ↔ output projection |
| Tokenizer | custom BPE, 32,000 vocab, regex pre-tokenization |

Implementation notes:

- The shared expert fires on **every** token, which resists expert collapse far better than
  load-balancing loss alone.
- Expert dispatch uses gather → batched per-expert forward → `index_add_` scatter. This is a
  deliberate simplification: a grouped GEMM (`ponytail:` ceiling) would cut the per-step Python
  loop and its GPU synchronizations, but the loop runs only 4 experts × 12 layers and was not
  worth the correctness risk.

## Training recipe

**Phase 1 — Alpaca warmup.** 600 steps on Alpaca, reaching val 3.85. This phase **overfits**
after step 600 (val rises to 5.8 by step 1400) and exists only to give Phase 2 a sane
initialization.

**Phase 2 — continued pre-training.** Resumed from the Phase-1 checkpoint on a streamed
FineWeb-Edu + Cosmopedia-v2 corpus (4:1 mix), encoded with the frozen 32k tokenizer:

| Split | Tokens |
|---|---|
| train | 180,295,081 |
| val | 20,109,016 |

Optimisation: AdamW (lr 1e-3 → 1e-4 cosine, 200 warmup steps, weight decay 0.1), grad clip 1.0,
global batch **32,768 tokens** (2 ranks × 16 grad-accum × 1 micro-batch × 1024 ctx), fp16 with
`GradScaler` (T4 has no bf16), 10-hour wall-clock cap per Kaggle session.

### Engineering notes

- **FSDP FULL_SHARD** was required to fit at all: 1B parameters + AdamW moments ≈ 14 GB barely
  exceeds one 15 GB T4, so optimizer state is sharded across both GPUs.
- **NCCL deadlock fix.** A full-state-dict gather is a collective. Rank 0 deciding whether to
  save while other ranks skip the gather hangs the job until the 10-minute timeout — the save
  decision is now broadcast so every rank enters the collective together.
- **Gradient-sync fix.** `no_sync()` on non-final micro-steps avoids 15 redundant all-reduces
  per optimizer step.
- **Domain-shift bookkeeping.** Phase 1 and Phase 2 validation losses are on *different datasets*
  and are not comparable. Phase-1's stale 3.85 was carried inside every early Phase-2 checkpoint
  and silently blocked all `best_model.pth` saves; best-tracking is now reset when seeding from a
  Phase-2 backup, and checkpoints record `phase2_best_val_loss` / `latest_val_loss` separately.

## Sample output

Base LM, no instruction tuning. Same prompt, before and after the final epoch.

**Step 3,365 (val 4.48)**

```
The theory of relativity was developed by Marx–Marx.
This theory is based on a theory of wave theory, in which the theory of relativity
and the classical theory of relativity were founded, but the theory of classical
mechanics was soon developed.
```

**Step 5,499 (val 3.96)**

```
The theory of relativity was developed by Robert Kriva and James Kriva in the 19th
century and was described by J. W. Schmidt in 1986 by Daniel L. Kriva and E. N. Eckert
(1995). A fundamental theory of relativity is a theory of relativity, developed by
James Kriva in the 1960s and later by Ralph Eckert, who introduced the concept of
relativity in the 1970s.
```

The final epoch removed the degenerate repetition loops and produced full grammatical clauses with
academic register. **Factual recall is still absent** — see Limitations.

## Limitations

Honest scope of what this model is:

- **No factual knowledge.** At 0.176 tokens/param it learned fluency and style, not facts.
  Compute-optimal for 1B is ~20 tokens/param (~20B tokens); this run is data-limited by design.
  Generated names are plausible and wrong.
- **Not instruction-tuned.** Trained only on web pages and textbook prose, so it continues
  documents rather than answering questions. A short SFT phase on instruction data would teach the
  *format* of answering — not the truth.
- **Short context.** 1024 tokens.
- **Optimizer state was not checkpointed across sessions.** AdamW moments reset at each Kaggle
  session boundary (~4 times). This costs some convergence but is not the limiting factor.
  The upgrade path is sharded FSDP optimizer-state serialization.
- **Single-node scale.** Two GPUs over PCIe; no NVLink, so FSDP communication is the throughput
  bottleneck (~1.4–1.8× headroom from `SHARD_GRAD_OP` + 8-bit AdamW was identified but not applied
  mid-run).

## Repository layout

```
.
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── architecture.py        # model definition (MoE, GQA, RoPE, sliding window, MTP)
├── tokenizer.py           # custom BPE (train / encode / decode / save / load)
├── data_loader.py         # memmap-backed batching over tokenized .bin files
├── prepare_data.py        # Phase-1 Alpaca tokenization
├── generate.py            # KV-cache text generation
├── train_custom.py        # FSDP training loop (the source of truth for the recipe)
└── notebooks/
    ├── 01_phase1_alpaca.ipynb
    ├── 02_phase2_pretrain.ipynb   # streams + encodes FineWeb-Edu / Cosmopedia-v2
    ├── 03_phase2_continue.ipynb   # self-resuming; seeds from newest backup
    ├── 04_test.ipynb              # generation smoke test
    └── 05_upload_hf.ipynb         # publish checkpoints to Hugging Face
```

Weights, tokenized `.bin` shards, and the tokenizer pickle are intentionally **not** in Git.
The tokenizer is retrained by `tokenizer.py`; the corpus is streamed from Hugging Face.

## Reproducing

Everything runs on Kaggle notebooks with **GPU T4 ×2** and **Internet ON** (only the data cells
need network).

1. Upload this repo's `.py` files as a Kaggle dataset (e.g. `kochi-llm/model`).
2. `notebooks/01_phase1_alpaca.ipynb` — Phase 1. Attach the dataset, Save & Run.
3. `notebooks/02_phase2_pretrain.ipynb` — attach the Phase-1 output, Save & Run. Streams and
   encodes ~180M tokens (~40 min).
4. `notebooks/03_phase2_continue.ipynb` — attach the previous run's output, Save & Run. This
   notebook is self-resuming: it picks the newest `latest_backup_model.pth` and restarts
   best-tracking, so the same notebook covers every subsequent session. Run it three times to
   reach step 5,500.
5. `notebooks/04_test.ipynb` — attach the final output to generate samples.

Each continuation session is capped at 10 hours and writes `latest_backup_model.pth`; the final
session exits the loop normally and also writes the endpoint checkpoint.

## Publishing weights

`notebooks/05_upload_hf.ipynb` publishes the model to **`<hf-username>/KochiLLM-v1.0`**.

It converts the most-trained checkpoint of **every attached run** to bf16 safetensors (~2.05 GB) and
pushes each as its own git revision tagged `run-NN`, with `metadata.json` recording step count,
validation loss, and source run. `main` always holds the highest-step checkpoint, and is only
overwritten by a higher step count — so the notebook is order-independent and safe to re-run.

Each checkpoint is 4.1 GB on disk, so attaching all nine at once (~37 GB) will likely exceed Kaggle's
notebook input limit. Attach as many as the limit allows, run the notebook, then attach the next
batch and run it again.

The release is deliberately **not** an `AutoModel` package: the tokenizer is a custom pickle and the
architecture is custom code, so the repo ships `architecture.py`, `tokenizer.py`, `generate.py`, the
tokenizer pickle, `config.json`, and `model.safetensors`, plus load instructions in the model card.
Embedding and output projection are tied, so `output.weight` is omitted from the safetensors file
(recorded in `metadata.json`) and restored on load.

## License

MIT — see [LICENSE](LICENSE).
