#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import pandas as pd
from datasets import Dataset, load_metric
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          TrainingArguments, Trainer, DataCollatorForTokenClassification)
from utils import read_csv_try_json, align_labels_with_tokens, get_label_mappings

def tokenize_and_align(examples, tokenizer, label_to_id):
    tokenized_inputs = tokenizer(examples["tokens_parsed"],
                                 is_split_into_words=True,
                                 truncation=True,
                                 max_length=512,
                                 padding="max_length")
    labels = []
    for i in range(len(examples["labels_parsed"])):
        word_labels = examples["labels_parsed"][i]
        # align
        aligned = align_labels_with_tokens(tokenized_inputs[i], examples["tokens_parsed"][i], word_labels, label_to_id)
        labels.append(aligned)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=False)
    parser.add_argument("--model_name", default="DFM/dfm-large", help="HuggingFace model id (replace with actual DFM-Large id)")
    parser.add_argument("--output_dir", default="models/dfm_large")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    # load data
    train_df = read_csv_try_json(args.train_csv)
    dfs = [train_df]
    label_list, label_to_id, id_to_label = get_label_mappings(dfs)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(label_list))

    # prepare datasets
    train_dataset = Dataset.from_pandas(train_df)
    # map to tokenized format
    def map_fn(ex):
        return tokenize_and_align(ex, tokenizer, label_to_id)
    tokenized_train = train_dataset.map(lambda ex: tokenize_and_align(ex, tokenizer, label_to_id), batched=True, remove_columns=train_dataset.column_names)

    data_collator = DataCollatorForTokenClassification(tokenizer)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        evaluation_strategy="no" if args.val_csv is None else "epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        logging_dir=os.path.join(args.output_dir, "logs"),
        load_best_model_at_end=False,
        save_total_limit=3
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    # save label mappings
    with open(os.path.join(args.output_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label_list": label_list, "label_to_id": label_to_id, "id_to_label": id_to_label}, f, ensure_ascii=False, indent=2)
    print("Training complete. Model saved to", args.output_dir)

if __name__ == "__main__":
    main()