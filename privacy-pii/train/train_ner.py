# train/train_ner.py
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from datasets import Dataset, DatasetDict

# ---- 프로젝트 루트 경로를 PYTHONPATH에 추가 ----
FILE_DIR = Path(__file__).resolve().parent
PROJ_DIR = FILE_DIR.parent
sys.path.insert(0, str(PROJ_DIR))       # privacy-pii/
sys.path.insert(0, str(FILE_DIR))       # privacy-pii/train/

# ---- 내부 유틸 임포트 ----
from train.data_utils import (
    read_bio_tsv,
    to_hf_dataset,
    build_label_maps,
    tokenize_and_align_labels,
)

# =========================
# 스키마 로더 (두 포맷 모두 지원)
# =========================
def load_labels(schema_path: str) -> List[str]:
    """
    1) {"labels": ["O","B-이름",...]}  형식
    2) {"O":0,"B-이름":1,...}         형식  → id 오름차순으로 정렬
    3) ["O","B-이름",...]             형식
    읽기 실패 시 ["O"] 반환
    """
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "labels" in obj:
            labels = obj["labels"]
        elif isinstance(obj, dict):
            items = [(lab, int(i)) for lab, i in obj.items()]
            items.sort(key=lambda x: x[1])
            labels = [lab for lab, _ in items]
        elif isinstance(obj, list):
            labels = obj
        else:
            labels = ["O"]
    except Exception:
        print(f"[WARN] failed to read schema at {schema_path}; fallback to ['O']")
        labels = ["O"]
    if "O" not in labels:
        labels = ["O"] + labels
    # 중복 제거(순서 유지)
    seen = set(); ordered = []
    for x in labels:
        if x not in seen:
            seen.add(x); ordered.append(x)
    return ordered

# =========================
# 데이터 준비
# =========================
def build_datasets(train_tsv: str, valid_tsv: str | None) -> DatasetDict:
    train_sents = read_bio_tsv(train_tsv)
    train_ds: Dataset = to_hf_dataset(train_sents)
    if valid_tsv and Path(valid_tsv).exists():
        valid_sents = read_bio_tsv(valid_tsv)
        valid_ds: Dataset = to_hf_dataset(valid_sents)
    else:
        split = train_ds.train_test_split(test_size=0.05, seed=42)
        train_ds, valid_ds = split["train"], split["test"]
    return DatasetDict(train=train_ds, validation=valid_ds)

def extract_labels_from_dataset(dsd: DatasetDict) -> List[str]:
    """데이터셋에 실제로 등장한 BIO 라벨 수집 (O 포함)."""
    tags = set()
    for split in dsd:
        for row in dsd[split]:
            for t in row["ner_tags"]:
                if t is None:
                    continue
                tag = (t if isinstance(t, str) else str(t)).strip()
                if tag:
                    tags.add(tag)
    tags = sorted(tags)
    # O가 있으면 맨 앞으로
    if "O" in tags:
        tags.remove("O")
        tags = ["O"] + tags
    return tags or ["O"]

def ensure_schema_covers_data(schema_labels: List[str], data_labels: List[str]) -> List[str]:
    """스키마에 없는 라벨을 데이터에 맞춰 자동 보강."""
    set_schema = set(schema_labels)
    missing = [t for t in data_labels if t not in set_schema]
    if missing:
        print(f"[WARN] schema is missing labels found in data: {missing}")
        schema_labels = schema_labels + [t for t in missing if t not in set_schema]
    # 중복 제거(순서 유지)
    seen = set(); ordered = []
    for x in schema_labels:
        if x not in seen:
            seen.add(x); ordered.append(x)
    return ordered

# =========================
# metric (선택적) — seqeval 없으면 생략
# =========================
def build_compute_metrics(id2label: Dict[int, str]):
    try:
        import evaluate  # pip install evaluate seqeval
        seqeval = evaluate.load("seqeval")
    except Exception:
        seqeval = None

    def _compute(pred):
        if seqeval is None:
            return {}
        logits, labels = pred
        preds = np.argmax(logits, axis=-1)

        true_labels, true_preds = [], []
        for p, l in zip(preds, labels):
            tl, tp = [], []
            for pi, li in zip(p, l):
                if li == -100:
                    continue
                tl.append(id2label[int(li)])
                tp.append(id2label[int(pi)])
            true_labels.append(tl)
            true_preds.append(tp)
        results = seqeval.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": results.get("overall_precision", 0.0),
            "recall": results.get("overall_recall", 0.0),
            "f1": results.get("overall_f1", 0.0),
            "accuracy": results.get("overall_accuracy", 0.0),
        }
    return _compute

# =========================
# 메인 학습 루프
# =========================
def main():
    p = argparse.ArgumentParser(description="Train NER (BIO) with HuggingFace Transformers")
    p.add_argument("--model_name", type=str, default="klue/bert-base")
    p.add_argument("--schema", type=str, default=str(PROJ_DIR / "config" / "labels_schema.json"))
    p.add_argument("--train_tsv", type=str, required=True)
    p.add_argument("--valid_tsv", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=str(PROJ_DIR / "outputs" / "ner"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 데이터셋부터 구성 (데이터 내 라벨 추출 위해)
    dsd = build_datasets(args.train_tsv, args.valid_tsv)

    # 2) 스키마 라벨 로드 + 데이터 라벨로 자동 보강
    labels = load_labels(args.schema)
    data_labels = extract_labels_from_dataset(dsd)
    labels = ensure_schema_covers_data(labels, data_labels)
    label2id, id2label = build_label_maps(labels)
    num_labels = len(labels)
    print(f"[INFO] num_labels={num_labels}")
    print(f"[INFO] labels={labels}")

    # 3) 토크나이저 & 라벨 정렬
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def _tok_fn(examples: Dict[str, Any]):
        return tokenize_and_align_labels(examples, tokenizer, label2id, max_length=args.max_length)

    dsd_tokenized = dsd.map(_tok_fn, batched=True, remove_columns=["tokens", "ner_tags"])
    dsd_tokenized.set_format(type="torch")

    # 4) 모델
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # 5) 콜레이터
    collator = DataCollatorForTokenClassification(tokenizer)

    # 6) 트레이닝 세팅
    total_train_steps = max(1, (len(dsd_tokenized["train"]) // max(1, args.batch_size)) * args.epochs)
    warmup_steps = int(total_train_steps * args.warmup_ratio)

    try:
    # 최신/표준 API (transformers 4.x 권장)
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=warmup_steps,
            evaluation_strategy="steps",   # ← 구버전에선 없는 인자
            eval_steps=args.eval_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=args.fp16,
            report_to=[],  # wandb 등 리포터 비활성
        )
    except TypeError as e:
        # 아주 구버전 호환 (evaluation_strategy 등 제거)
        print("[WARN] TrainingArguments new API not available, falling back:", e)
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=warmup_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=args.fp16,
            # 구버전에선 자동평가/베스트모델 로딩이 비활성일 수 있음
        )

    # 7) 트레이너
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dsd_tokenized["train"],
        eval_dataset=dsd_tokenized["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(id2label),
    )

    # 8) 학습
    trainer.train()

    # 9) 저장
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(Path(args.output_dir) / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    # 10) 최종 평가
    metrics = trainer.evaluate()
    print("Eval:", metrics)

if __name__ == "__main__":
    main()
