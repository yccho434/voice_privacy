# train/train_ner.py
import os, json, math, random
import argparse, yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from seqeval.metrics import classification_report, f1_score

from nlp.datasets import BIOTokenDataset, load_schema
from nlp.modeling import build_model

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); 
    torch.cuda.manual_seed_all(seed)

def decode_predictions(pred_ids, labels_mask, id2label):
    # pred_ids: list[List[int]] if CRF, else tensor [B, L]
    out = []
    if isinstance(pred_ids, list):
        # already per-sample lengths
        for seq in pred_ids:
            out.append([id2label[i] for i in seq])
    else:
        pred_ids = pred_ids.cpu().numpy()
        for row, mask in zip(pred_ids, labels_mask):
            seq=[]
            for i, m in zip(row, mask):
                if not m: continue
                seq.append(id2label[int(i)])
            out.append(seq)
    return out

def decode_gold(labels, labels_mask, id2label):
    out = []
    labels = labels.cpu().numpy()
    for row, mask in zip(labels, labels_mask):
        seq=[]
        for i, m in zip(row, mask):
            if not m: continue
            seq.append(id2label[int(i)])
        out.append(seq)
    return out

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    set_seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() and cfg["device"]=="cuda" else "cpu"

    label2id, id2label = load_schema(cfg["labels"]["schema_path"])
    num_labels = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_model_name"])
    train_ds = BIOTokenDataset(cfg["data"]["train_path"], tokenizer, label2id, cfg["data"]["max_length"])
    valid_ds = BIOTokenDataset(cfg["data"]["valid_path"], tokenizer, label2id, cfg["data"]["max_length"])

    model = build_model(cfg["hf_model_name"], num_labels, id2label, label2id, cfg["use_crf"]).to(device)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    no_decay = ["bias", "LayerNorm.weight"]
    optim_params = [
        {"params":[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": cfg["train"]["weight_decay"]},
        {"params":[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optim_params, lr=cfg["train"]["lr"])
    total_steps = math.ceil(len(train_loader)*cfg["train"]["epochs"]/cfg["train"]["grad_accum_steps"])
    warmup_steps = int(total_steps * cfg["train"]["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["fp16"])

    best_f1 = -1.0
    os.makedirs(cfg["train"]["save_dir"], exist_ok=True)

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        total_loss=0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k:v.to(device) for k,v in batch.items()}
            with torch.cuda.amp.autocast(enabled=cfg["train"]["fp16"]):
                out = model(**batch)
                loss = out["loss"] if isinstance(out, dict) else out.loss
            scaler.scale(loss).backward()
            if step % cfg["train"]["grad_accum_steps"] == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(); scheduler.step()
            total_loss += loss.item()
        # eval
        model.eval()
        all_preds, all_golds = [], []
        with torch.no_grad():
            for batch in valid_loader:
                labels = batch["labels"].to(device)
                attn = batch["attention_mask"].to(device)
                mask = (labels!=-100).cpu().numpy().astype(bool)
                batch = {k:v.to(device) for k,v in batch.items()}
                out = model(**{k:v for k,v in batch.items() if k!="labels"})
                if "pred_ids" in out:
                    preds = out["pred_ids"]
                else:
                    preds = out["logits"].argmax(-1)
                pred_tags = decode_predictions(preds, mask, id2label)
                gold_tags = decode_gold(labels, mask, id2label)
                all_preds.extend(pred_tags); all_golds.extend(gold_tags)

        f1 = f1_score(all_golds, all_preds)
        print(f"[epoch {epoch+1}] loss={total_loss/len(train_loader):.4f}  f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            save_dir = cfg["train"]["save_dir"]
            model_to_save = model
            torch.save(model_to_save.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
            tokenizer.save_pretrained(save_dir)
            with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump({"id2label": id2label, "label2id": label2id, "use_crf": cfg["use_crf"]}, f, ensure_ascii=False)
            print(f"** saved best to {save_dir} (f1={best_f1:.4f})")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/pii_ner.yaml")
    args = parser.parse_args()
    main(args.cfg)
