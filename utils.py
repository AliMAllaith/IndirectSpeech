import json
import ast
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import PreTrainedTokenizerBase

def read_csv_try_json(path: str) -> pd.DataFrame:
    """
    Read CSV and try to parse 'tokens' and 'labels' if they are JSON-like.
    Supported columns:
      - tokens (JSON list) OR text (string)
      - labels (JSON list) OR labels (space-separated string)
    """
    df = pd.read_csv(path)
    # normalize column names
    cols = {c.lower(): c for c in df.columns}
    # prefer 'tokens' and 'labels'
    if 'tokens' in cols:
        # try to parse JSON lists
        def parse_cell(x):
            if pd.isna(x):
                return None
            try:
                if isinstance(x, str) and x.strip().startswith('['):
                    return json.loads(x)
                return ast.literal_eval(x) if isinstance(x, str) else x
            except Exception:
                # fallback to whitespace split
                return str(x).split()
        df['tokens_parsed'] = df[cols['tokens']].apply(parse_cell)
    elif 'text' in cols:
        df['text'] = df[cols['text']].astype(str)
        df['tokens_parsed'] = df['text'].apply(lambda t: t.split())
    else:
        raise ValueError("CSV must contain either 'tokens' or 'text' column.")

    if 'labels' in cols:
        def parse_labels(x):
            if pd.isna(x):
                return None
            try:
                if isinstance(x, str) and x.strip().startswith('['):
                    return json.loads(x)
                return str(x).split()
            except Exception:
                return str(x).split()
        df['labels_parsed'] = df[cols['labels']].apply(parse_labels)
    else:
        raise ValueError("CSV must contain a 'labels' column (JSON list or space-separated labels).")
    return df[['tokens_parsed', 'labels_parsed']]

def align_labels_with_tokens(tokenized_inputs, words, labels, label_to_id):
    """
    Align labels (one per word in 'words') with tokenized inputs (using is_split_into_words=True tokenization).
    Returns input labels aligned with tokenization, using -100 for subword tokens to be ignored by loss.
    """
    word_ids = tokenized_inputs.word_ids(batch_index=0)
    aligned_labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            # label for the first token of the word
            aligned_labels.append(label_to_id[labels[word_idx]])
        else:
            # For subsequent tokens of a word, set -100 to ignore or repeat label depending on strategy
            aligned_labels.append(-100)
        previous_word_idx = word_idx
    return aligned_labels

def get_label_mappings(dfs):
    """
    Get sorted list of labels from list of dataframes with 'labels_parsed' column
    """
    all_labels = set()
    for df in dfs:
        for labs in df['labels_parsed'].dropna():
            all_labels.update(labs)
    labels = sorted(list(all_labels))
    label_to_id = {l: i for i,l in enumerate(labels)}
    id_to_label = {i: l for l,i in label_to_id.items()}
    return labels, label_to_id, id_to_label