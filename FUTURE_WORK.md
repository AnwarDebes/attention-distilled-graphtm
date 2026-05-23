# Future Work: Decoder-Only LM Attention Distillation

Extending the structural-distillation recipe from BERT to decoder-only
language models (Qwen, LLaMA, Mistral, Gemma).

> University of Agder (UiA)
> Status: planned follow-on (R8 baseline: 94.77% mean across 5 seeds).

## Premise

The current work distilled BERT's attention into the GraphTM's graph
topology and recovered most of BERT's R8 accuracy in a pure-Boolean
clause learner. The student-side machinery is teacher-agnostic and
consumes nodes and edges, nothing else. This plan answers: does the
same trick transfer to decoder-only LMs, and does the richer
relational structure of a larger modern LM translate into a stronger
Boolean student?

## Hypothesis

1. Causal attention (Qwen, LLaMA) yields a lower-triangular DAG
   rather than BERT's bidirectional graph. The student still
   benefits, but the inductive bias differs and may favour different
   datasets.
2. Scaling the teacher (1.5B to 7B to 14B) gives diminishing returns
   once attention patterns saturate. A crossover exists past which
   the extraction cost outweighs the gain.
3. Task-aligned teachers (LoRA-fine-tuned on the downstream task)
   beat raw base or instruct teachers, mirroring the reliance on a
   classification-fine-tuned BERT.

## Scope

In scope: R8 first, then R52, 20NG, Ohsumed, IMDb. Teachers
Qwen2.5-{1.5B, 3B, 7B}, LLaMA-3.2-{1B, 3B}, and Mistral-7B-Instruct
as a cross-family control.

Out of scope: generative tasks (the student is a classifier), models
that don't expose per-head attention via HuggingFace, and teachers
above 14B unless the layer-sweep results justify the compute.

## Plan

### Phase 1: Baseline transfer (Qwen2.5-1.5B, R8)

LoRA-fine-tune Qwen2.5-1.5B-Instruct on R8 with a classification head
(QLoRA: 4-bit base, rank 16, lr 2e-4, 3 epochs, AdamW). Save under
`$QWEN_TEACHER_DIR/qwen2p5_1p5b_teacher_R8/seed_42/`.

Load with `attn_implementation="eager", output_attentions=True,
torch_dtype=torch.bfloat16`. The FlashAttention and SDPA paths return
`None` for attentions, so eager is mandatory.

Mirror the BERT setup: `top_k=5`, layers `[10, 14, 18]` (mid-to-late
of the 28-layer stack), average across heads.

Account for GQA. Qwen2.5-1.5B has 12 Q heads / 2 KV heads. Averaging
the per-Q-head attention matrices works unchanged, just note that
fewer patterns are independent.

Reuse `experiments/train_paper_b_attention_distill.py` end-to-end
with the teacher loader swapped. Target: original BERT result within
1 pp.

### Phase 2: Layer-selection sweep

Sweep `layers` over `{[6,10,14], [10,14,18], [14,18,22], [18,22,26],
all-layers-pooled}` and `top_k` over `{3, 5, 8, 12}`. Three seeds per
cell to control cost, full 5-seed runs only for the winning
configuration. Expectation: mid-to-late layers dominate, matching the
BERT 6/8/10 finding.

### Phase 3: Teacher scaling ladder

| Teacher | Params | Extraction cost (R8 train, 1x A100) | Expected gain over Phase 1 |
|---|---|---|---|
| Qwen2.5-1.5B | 1.5B | ~10 min | baseline |
| Qwen2.5-3B | 3B | ~25 min | +0.3 to 0.7 pp |
| Qwen2.5-7B | 7B | ~70 min | +0.5 to 1.0 pp |
| Qwen2.5-14B | 14B | ~3 h | likely saturated |

Stop scaling when two consecutive steps yield less than 0.3 pp at the
same seeds.

### Phase 4: Causal vs. bidirectional ablation

Same dataset (R8), same student hyperparameters, three teachers:

- BERT-base (bidirectional, original baseline).
- Qwen2.5-1.5B (causal, comparable order-of-magnitude param count).
- DeBERTa-v3-base (bidirectional, modern encoder control).

Report attention-graph statistics alongside student accuracy:

- Average in/out degree per node.
- Edge symmetry ratio (fraction of edges with a reverse counterpart).
- Longest dependency path and graph diameter.
- Per-token attention entropy (sanity check for "too diffuse").

The scientific question: is bidirectional structure intrinsically
more useful for clause learning, or does scale of a causal teacher
compensate?

### Phase 5: Harder datasets

Take the Phase 1 winner and run on R52, 20NG, Ohsumed, IMDb using the
BertGCN canonical splits loaded via `src/utils/load_bertgcn_splits.py`.
Track where the causal vs. bidirectional gap widens or narrows.
Longer documents (IMDb) are where causal attention has the most room
to encode useful long-range structure.

## Engineering specifics

Attention extraction loads the teacher with
`AutoModelForSequenceClassification.from_pretrained(path,
attn_implementation="eager", output_attentions=True,
torch_dtype=torch.bfloat16)`. Verify `outputs.attentions is not None`
before the first batch.

Causal attention is zero in the upper triangle. The top-k extraction
at `experiments/train_paper_b_attention_distill.py:108` naturally
restricts to valid (lower-triangular) targets, so no code change is
needed. Side effect: early tokens have very few candidates while
later tokens have many. Log edges-per-doc histograms; if the head of
the sequence ends up edge-starved, consider a position-conditional
`top_k`.

GQA does not break head averaging. Per-Q-head attention is exposed
even when keys and values are shared across query heads, so
`attentions[l][b].mean(dim=0)` works as-is.

Qwen BPE differs from BERT WordPiece, but the GraphTM treats tokens
as opaque symbols, so semantics do not need to match. The vocabulary
will differ. Keep the top-5000-tokens cutoff used in
`extract_attention_graphs`.

The training code uses `max_len=128`. That covers R8 (mean ~70
tokens). Bump to 256 for IMDb and 512 for 20NG long documents.

Do not wrap inputs in chat templates when using `*-Instruct`
teachers. Attention concentrates on `<|im_start|>` scaffolding and
pollutes the structural signal. Feed raw text directly.

## Evaluation methodology

Match the original rigour:

- 5 seeds per configuration (42, 123, 456, 789, 1337).
- Paired Wilcoxon signed-rank tests via
  `src/eval/stats.py:paired_wilcoxon`, alternative `"greater"` when
  claiming one method beats another.
- Bootstrap 95% CI via `bootstrap_ci`, n=1000 resamples.
- Report mean +/- std and median. Never headline the best seed.
- Bonferroni-correct across multi-comparison tables.

## Compute budget estimate

| Phase | GPU-hours (A100) | Wall clock (1x A100) |
|---|---|---|
| 1 | 4 | half day |
| 2 | 20 | 2 to 3 days |
| 3 | 30 | 3 days |
| 4 | 8 | 1 day |
| 5 | 25 | 2 to 3 days |
| Total | ~85 GPU-h | ~10 days |

Halve the wall-clock with 2x A100 by parallelising seeds. The
extraction step is embarrassingly parallel per document.

## Risks and open questions

Causal attention may be too sparse on short documents. A 30-token R8
headline has 30 * 29 / 2 ~= 435 valid edges under causal masking vs.
870 bidirectional, so `top_k=5` may saturate the available targets
near the head of the sequence. Mitigation: log edges-per-doc
histograms and consider a position-conditional `top_k`.

Instruction-tuned models attend to chat scaffolding (already noted
under engineering specifics). Inspect attention heatmaps on the first
20 documents before launching a full sweep.

Large LMs often produce very flat attention distributions, and top-k
can then pick up noise. Sanity check: per-token attention entropy vs.
a small-LM baseline. If entropy is high and accuracy disappoints, try
a temperature-style top-k that requires the k-th edge to be at least
tau times the 1st edge weight.

LoRA fine-tuning needs to be reproducible. Pin LoRA hyperparameters
and data ordering, and hash the resulting adapter weights into the
`dataset_hashes` field of `ExperimentLogger.log_config` so seed
comparisons are honest.

## Deliverables

- `experiments/train_paper_c_qwen_distill.py`: Qwen teacher loader,
  otherwise structurally identical to the existing training script.
- `experiments/paper_c_qwen_R8/seed_{42,123,456,789,1337}/results.json`:
  same schema as the existing per-seed results.
- `results/paper_c_qwen_R8_5seeds.json`: aggregated 5-seed summary.
- `MODELS.md`: append a "Qwen teacher" section with the QLoRA recipe
  and `$QWEN_TEACHER_DIR` layout.
- Follow-on paper bundling the BERT and Qwen results as "structural
  distillation across teacher architectures". Venue TBD.

## Repo layout extension

```
attention-distilled-graphtm/
├── experiments/
│   ├── train_paper_b_attention_distill.py   # existing
│   ├── train_paper_c_qwen_distill.py        # new
│   ├── paper_b_attn_R8/                     # existing
│   └── paper_c_qwen_R8/seed_{42,...}/       # new
├── results/
│   ├── paper_b_R8_5seeds.json               # existing
│   └── paper_c_qwen_R8_5seeds.json          # new
└── src/
    └── teachers/                             # new
        ├── bert_attention.py                 # refactored out of experiments/
        └── qwen_attention.py                 # new
```

The only structural change is refactoring `extract_attention_graphs`
out of `train_paper_b_attention_distill.py` into
`src/teachers/bert_attention.py`, so both teachers expose the same
`extract_attention_graphs(teacher, ...)` interface and stay
swappable.
