#!/usr/bin/env python3
"""Attention-Distilled Graph Tsetlin Machine.

Use BERT's attention edges as GraphTM graph topology. The student is
a pure clause-based classifier.

Steps:
1. Load fine-tuned BERT checkpoint
2. For each document, extract top-k attention edges from BERT
3. Build GraphTM graphs with attention-derived topology
4. Train GraphTM on these graphs
"""

import os, sys, json, time, pickle
import numpy as np
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from eval.logger import ExperimentLogger


def extract_attention_graphs(texts, labels, model_path, tokenizer_name,
                              top_k=5, layers=[6, 8, 10], max_len=128, batch_size=32):
    """Extract attention-based graph structure from BERT for all documents."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, output_attentions=True
    ).to(device).eval()

    all_nodes = []
    all_edges = []

    # Collect all unique tokens for vocab
    token_counts = Counter()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        inputs = tokenizer(
            batch_texts, return_tensors="pt", truncation=True,
            max_length=max_len, padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Process each document in batch
        for b in range(len(batch_texts)):
            seq_len = int(attention_mask[b].sum())
            tokens = tokenizer.convert_ids_to_tokens(input_ids[b][:seq_len])

            # Average attention over selected layers and heads
            layer_attns = []
            for l in layers:
                if l < len(attentions):
                    # (heads, seq, seq) -> average over heads -> (seq, seq)
                    layer_attns.append(attentions[l][b].mean(dim=0)[:seq_len, :seq_len].cpu().numpy())

            if not layer_attns:
                continue

            avg_attn = np.mean(layer_attns, axis=0)  # (seq, seq)

            # Build nodes (skip [CLS] and [SEP])
            nodes = []
            token_to_idx = {}
            for t_idx in range(1, len(tokens) - 1):  # skip CLS/SEP
                tok = tokens[t_idx]
                node_name = f"{tok}_{len(nodes)}"
                nodes.append((node_name, tok))
                token_to_idx[t_idx] = len(nodes) - 1
                token_counts[tok] += 1

            if not nodes:
                nodes = [("[UNK]_0", "[UNK]")]
                token_to_idx[1] = 0

            # Extract top-k attention edges
            edges = []
            for src_orig in range(1, len(tokens) - 1):
                if src_orig not in token_to_idx:
                    continue
                src_idx = token_to_idx[src_orig]

                weights = avg_attn[src_orig].copy()
                weights[0] = 0  # ignore CLS
                weights[src_orig] = 0  # ignore self
                if len(tokens) > 1:
                    weights[len(tokens)-1] = 0  # ignore SEP

                top_indices = np.argsort(weights)[::-1][:top_k]
                for dst_orig in top_indices:
                    if dst_orig in token_to_idx and weights[dst_orig] > 0:
                        dst_idx = token_to_idx[dst_orig]
                        edges.append((nodes[src_idx][0], nodes[dst_idx][0], "attn"))

            # Also add sequential edges
            for j in range(len(nodes) - 1):
                edges.append((nodes[j][0], nodes[j+1][0], "seq"))
                edges.append((nodes[j+1][0], nodes[j][0], "seq_inv"))

            all_nodes.append(nodes)
            all_edges.append(edges)

        if (i // batch_size) % 50 == 0:
            print(f"  Extracted {min(i+batch_size, len(texts))}/{len(texts)} attention graphs")

    # Build vocab from most common tokens
    vocab = [tok for tok, _ in token_counts.most_common(5000)]

    return all_nodes, all_edges, vocab


def build_graphs(doc_nodes, doc_edges, vocab, hv_size=512, hv_bits=2, init_with=None):
    """Build GraphTM Graphs from attention-extracted nodes/edges."""
    num_graphs = len(doc_nodes)
    vocab_set = set(vocab)

    # Filter to vocab
    filtered_nodes = []
    filtered_edges = []
    for did in range(num_graphs):
        valid_names = set()
        fnodes = []
        for name, symbol in doc_nodes[did]:
            if symbol in vocab_set:
                fnodes.append((name, symbol))
                valid_names.add(name)
        if not fnodes:
            fnodes = [(f"{vocab[0]}_0", vocab[0])]
            valid_names = {fnodes[0][0]}
        fedges = [(s, d, e) for s, d, e in doc_edges[did]
                  if s in valid_names and d in valid_names]
        filtered_nodes.append(fnodes)
        filtered_edges.append(fedges)

    if init_with is not None:
        graphs = Graphs(num_graphs, init_with=init_with)
    else:
        graphs = Graphs(num_graphs, symbols=list(vocab),
                       hypervector_size=hv_size, hypervector_bits=hv_bits)

    for did in range(num_graphs):
        graphs.set_number_of_graph_nodes(did, len(filtered_nodes[did]))
    graphs.prepare_node_configuration()

    for did in range(num_graphs):
        ec = Counter()
        for s, d, e in filtered_edges[did]:
            ec[s] += 1
        for name, symbol in filtered_nodes[did]:
            graphs.add_graph_node(did, name, ec.get(name, 0))
    graphs.prepare_edge_configuration()

    for did in range(num_graphs):
        for s, d, e in filtered_edges[did]:
            graphs.add_graph_node_edge(did, s, d, e)

    for did in range(num_graphs):
        for name, symbol in filtered_nodes[did]:
            graphs.add_graph_node_property(did, name, symbol)

    graphs.encode()
    return graphs


def main():
    from datasets import load_dataset

    SEED = 42
    N_TRAIN = 5000  # Subset for speed
    N_TEST = 2000
    TOP_K = 5
    LAYERS = [6, 8, 10]
    CLAUSES = 5000
    T = 2500
    S = 5.0
    EPOCHS = 40

    exp_dir = os.path.join(PROJECT_ROOT, "experiments", "paper_b_attn_imdb", f"seed_{SEED}")
    os.makedirs(exp_dir, exist_ok=True)
    logger = ExperimentLogger(os.path.join(exp_dir, "log.jsonl"))

    # BERT teacher checkpoints were moved out of the repo. See MODELS.md.
    BERT_MODEL_DIR = os.environ.get("BERT_MODEL_DIR", os.path.expanduser("~/model_archive"))
    ckpt_dir = os.path.join(BERT_MODEL_DIR,
                            "baseline_bert-base-uncased_imdb", "seed_42", "checkpoints")
    ckpts = sorted(os.listdir(ckpt_dir), key=lambda x: int(x.split("-")[1]))
    bert_path = os.path.join(ckpt_dir, ckpts[-1])
    print(f"Using BERT checkpoint: {bert_path}")

    # Load IMDb
    print("Loading IMDb...")
    ds = load_dataset("imdb")
    train_texts = ds["train"]["text"][:N_TRAIN]
    train_labels = np.array(ds["train"]["label"][:N_TRAIN], dtype=np.uint32)
    test_texts = ds["test"]["text"][:N_TEST]
    test_labels = np.array(ds["test"]["label"][:N_TEST], dtype=np.uint32)
    print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")

    # Extract attention graphs from BERT
    print(f"\nExtracting attention graphs (top_k={TOP_K}, layers={LAYERS})...")
    t0 = time.time()
    train_nodes, train_edges, vocab = extract_attention_graphs(
        train_texts, train_labels, bert_path, "bert-base-uncased",
        top_k=TOP_K, layers=LAYERS
    )
    t_extract_train = time.time() - t0
    print(f"Train extraction: {t_extract_train:.1f}s")

    print("Extracting test attention graphs...")
    t0 = time.time()
    test_nodes, test_edges, _ = extract_attention_graphs(
        test_texts, test_labels, bert_path, "bert-base-uncased",
        top_k=TOP_K, layers=LAYERS
    )
    t_extract_test = time.time() - t0
    print(f"Test extraction: {t_extract_test:.1f}s")

    avg_nodes = np.mean([len(n) for n in train_nodes])
    avg_edges = np.mean([len(e) for e in train_edges])
    print(f"Avg nodes/doc: {avg_nodes:.1f}, Avg edges/doc: {avg_edges:.1f}")
    print(f"Vocab: {len(vocab)} tokens")

    # Build GraphTM graphs
    print("\nBuilding GraphTM graphs...")
    t0 = time.time()
    graphs_train = build_graphs(train_nodes, train_edges, vocab, hv_size=512, hv_bits=2)
    t_build = time.time() - t0
    print(f"Train graphs: {t_build:.1f}s")

    graphs_test = build_graphs(test_nodes, test_edges, vocab, init_with=graphs_train)
    print("Test graphs built")

    # Train GraphTM
    print(f"\nTraining GraphTM: clauses={CLAUSES}, T={T}, s={S}, depth=1")
    tm = MultiClassGraphTsetlinMachine(
        CLAUSES, T, S, depth=1,
        message_size=256, message_bits=2,
        max_included_literals=32,
    )

    logger.log_config({
        "method": "attention-distilled GraphTM",
        "bert_checkpoint": bert_path,
        "top_k": TOP_K, "layers": LAYERS,
        "clauses": CLAUSES, "T": T, "s": S,
        "n_train": N_TRAIN, "n_test": N_TEST,
    })

    best_acc = 0.0
    for epoch in range(EPOCHS):
        t0 = time.time()
        tm.fit(graphs_train, train_labels, epochs=1, incremental=True)
        t_train = time.time() - t0

        preds = tm.predict(graphs_test)
        acc = 100 * (preds == test_labels).mean()
        if acc > best_acc:
            best_acc = acc

        logger.log_epoch(epoch, {
            "accuracy": float(acc),
            "best_accuracy": float(best_acc),
            "train_time": float(t_train),
        })

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"  E{epoch:2d}: acc={acc:.2f}% (best={best_acc:.2f}%) time={t_train:.1f}s")

    print(f"\n=== Paper B: Attention-Distilled GraphTM on IMDb = {best_acc:.2f}% ===")
    print(f"Comparison: TM BoW = 89.98%, BERT = 92.24%")

    summary = {
        "experiment": "paper_b_attn_imdb",
        "method": "Attention-Distilled GraphTM",
        "seed": SEED,
        "best_test_accuracy": float(best_acc),
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "comparisons": {
            "tm_bow": 89.98,
            "bert_base": 92.24,
        },
    }

    logger.log_summary(summary)
    logger.close()

    with open(os.path.join(exp_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {exp_dir}/summary.json")


if __name__ == "__main__":
    main()
