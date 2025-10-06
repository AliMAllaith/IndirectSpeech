#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import pandas as pd
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from utils import read_csv_try_json, align_labels_with_tokens, get_label_mappings

def tokenize_and_align_single(example, tokenizer, label_to_id):
    tokenized = tokenizer(example["tokens_parsed"], is_split_into_words=True, truncation=True, max_length=512, padding="max_length")
    aligned = align_labels_with_tokens(tokenized, example["tokens_parsed"], example["labels_parsed"], label_to_id)
    tokenized["labels"] = aligned
    return tokenized

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output", default="report_test.json")
    args = parser.parse_args()

    test_df = read_csv_try_json(args.test_csv)
    with open(os.path.join(args.model_dir, "label_map.json"), "r", encoding="utf-8") as f:
        label_map = json.load(f)
    label_list = label_map["label_list"]
    label_to_id = label_map["label_to_id"]
    id_to_label = label_map["id_to_label"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)

    # prepare dataset
    test_ds = Dataset.from_pandas(test_df)
    tokenized_test = test_ds.map(lambda ex: tokenize_and_align_single(ex, tokenizer, label_to_id), remove_columns=test_ds.column_names)

    # Run predictions
    from transformers import Trainer, TrainingArguments
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(model=model, args=TrainingArguments(output_dir="tmp"), data_collator=data_collator, tokenizer=tokenizer)
    preds_output = trainer.predict(tokenized_test)
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids

    # convert to label strings, ignoring -100
    preds_list = []
    gold_list = []
    for i in range(len(labels)):
        pred_seq = []
        gold_seq = []
        for p, l in zip(preds[i], labels[i]):
            if l == -100:
                continue
            pred_seq.append(label_list[p])
            gold_seq.append(label_list[l])
        preds_list.append(pred_seq)
        gold_list.append(gold_seq)

    # compute classification report with sklearn
    from sklearn.metrics import classification_report
    # flatten
    y_true = [t for seq in gold_list for t in seq]
    y_pred = [p for seq in preds_list for p in seq]
    report = classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=True)
    # save report
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Saved classification report to", args.output)

if __name__ == "__main__":
    main()