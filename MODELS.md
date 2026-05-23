# BERT Teacher Checkpoints

The attention teacher is fine-tuned BERT-base-uncased. The checkpoint
is large (~4 GB), reproducible, and lives outside the repo.

## Default location

```
~/model_archive/
└── bert_teacher_R8/seed_42/
    └── checkpoint-<step>/   # the standard HuggingFace format
```

## Override

The training script reads `$BERT_MODEL_DIR`:

```bash
export BERT_MODEL_DIR=/path/to/your/teachers
python experiments/train_paper_b_attention_distill.py
```

## Reproducing the teacher

1. Load `bert-base-uncased` from HuggingFace.
2. Fine-tune on the R8 BertGCN split (see `data/precomputed_graphs/`
   for the same R8 used as student data; raw splits at
   `~/data_archive/bertgcn_splits/R8.txt` after the 2026-05-09 reorg).
3. Standard recipe: 3 epochs, lr=2e-5, batch=32, AdamW.
4. The trainer writes the checkpoint that the training script expects.

A reproduction script is left as future work. The current results were
obtained from the original training run preserved at
`~/model_archive/bert_teacher_R8/`.
