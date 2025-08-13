from __future__ import annotations
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import csv
from datasets import Dataset
import numpy as np

def load_schema(path: str) -> List[str]:
    """
    schema 파일(JSON/CSV 둘 다 지원). 
    - JSON: {"labels": ["O","B-이름","I-이름",...]}
    - CSV : header 없이 한 줄당 한 라벨
    """
    import json, os
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        labels = obj["labels"]
    else:
        labels = []
        with open(path, "r", encoding="utf-8") as f:
            for row in f:
                r = row.strip()
                if r:
                    labels.append(r)
    # O가 항상 0번이 되도록 정렬 보정 (없으면 추가)
    if "O" not in labels:
        labels = ["O"] + labels
    # 중복 제거 + 상대 순서 유지
    seen = set()
    ordered = []
    for x in labels:
        if x not in seen:
            seen.add(x)
            ordered.append(x)
    return ordered

def read_bio_tsv(path: str) -> List[Dict[str, Any]]:
    """
    BIO TSV 예: token \t tag
    문장 경계는 빈 줄.
    """
    sentences = []
    tokens, tags = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": tags})
                    tokens, tags = [], []
                continue
            parts = line.split("\t")
            if len(parts) == 1:
                tok, tag = parts[0], "O"
            else:
                tok, tag = parts[0], parts[1]
            tokens.append(tok)
            tags.append(tag)
    if tokens:
        sentences.append({"tokens": tokens, "ner_tags": tags})
    return sentences

def to_hf_dataset(train_sentences: List[Dict[str, Any]]) -> Dataset:
    return Dataset.from_list(train_sentences)

def build_label_maps(labels: List[str]) -> Tuple[Dict[str,int], Dict[int,str]]:
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label

def tokenize_and_align_labels(examples, tokenizer, label2id, max_length=256):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
    )
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word = None
        label_ids = []
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            else:
                label_str = (labels[wid] or "O").strip()
                if wid != previous_word:
                    label_ids.append(label2id.get(label_str, label2id["O"]))
                else:
                    if label_str.startswith("B-"):
                        label_str = label_str.replace("B-", "I-")
                    label_ids.append(label2id.get(label_str, label2id["O"]))
                previous_word = wid
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized
