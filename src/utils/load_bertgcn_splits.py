"""Load the canonical BertGCN train/test splits for R8, R52, 20NG, Ohsumed, MR.

These splits come from vendor/BertGCN/data/ (Lin et al., ACL Findings 2021).
Format: each .txt file has lines like "docname\t{train,test}\tlabel"
Clean text is in data/corpus/{dataset}.clean.txt (one doc per line, same order).
"""

import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_LOCATIONS = [
    os.environ.get("BERTGCN_DATA_DIR", ""),
    os.path.expanduser("~/data_archive/bertgcn_splits"),
    os.path.join(_REPO_ROOT, "bertgcn_splits"),
]
VENDOR_DIR = next((p for p in _DEFAULT_LOCATIONS if p and os.path.isdir(p)), _DEFAULT_LOCATIONS[1])

DATASETS = ["R8", "R52", "20ng", "ohsumed", "mr"]


def load_split(dataset):
    """Load a BertGCN dataset split.

    Args:
        dataset: one of "R8", "R52", "20ng", "ohsumed", "mr"

    Returns:
        dict with keys:
            train_texts, train_labels, test_texts, test_labels,
            label_names (sorted unique labels)
    """
    assert dataset in DATASETS, f"Unknown dataset: {dataset}. Choose from {DATASETS}"

    # Read split info
    split_path = os.path.join(VENDOR_DIR, f"{dataset}.txt")
    with open(split_path) as f:
        split_lines = [line.strip() for line in f]

    # Read clean text
    text_path = os.path.join(VENDOR_DIR, "corpus", f"{dataset}.clean.txt")
    with open(text_path) as f:
        texts = [line.strip() for line in f]

    # Parse splits
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for i, line in enumerate(split_lines):
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        doc_name, split_type, label = parts[0], parts[1], parts[2]

        if i < len(texts):
            text = texts[i]
        else:
            continue

        if "train" in split_type:
            train_texts.append(text)
            train_labels.append(label)
        elif "test" in split_type:
            test_texts.append(text)
            test_labels.append(label)

    label_names = sorted(set(train_labels + test_labels))
    label2idx = {l: i for i, l in enumerate(label_names)}

    return {
        "train_texts": train_texts,
        "train_labels": [label2idx[l] for l in train_labels],
        "test_texts": test_texts,
        "test_labels": [label2idx[l] for l in test_labels],
        "label_names": label_names,
        "label2idx": label2idx,
        "dataset": dataset,
        "n_train": len(train_texts),
        "n_test": len(test_texts),
        "n_classes": len(label_names),
    }
