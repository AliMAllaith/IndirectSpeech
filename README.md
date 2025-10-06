# Indirect Speech Fine-tuning 

Repository containing simple training and testing scripts for fine-tuning a transformer model on a word-level (token classification) Indirect Speech task.

## Structure
```
├── README.md
├── requirements.txt
├── train.py
├── test.py
├── utils.py
├── Models/        # saved checkpoints
└── Data/          # place train.csv and test.csv here
```

## Expected data format
`train.csv` and `test.csv` should be placed in `data/`. Each row should contain:
- `tokens`: a tokenized sentence (JSON array) **or** `text` (string)
- `labels`: a JSON array of token-level labels (same length as `tokens`) **or** a space-separated string of labels aligned to whitespace-tokenization

The `utils.py` loader will attempt to detect and parse either format. If you have plain text in `text`, token alignment uses a simple whitespace split to align labels; for best results, provide `tokens` + `labels`.

## Usage

Install dependencies:
```bash
pip install -r requirements.txt
```

Train (default model placeholder is `DFM/dfm-large` — replace with your actual model id or pass `--model_name`):
```bash
python train.py --train_csv data/train.csv --val_csv data/val.csv --output_dir models/dfm_large
```

Test:
```bash
python test.py --test_csv data/test.csv --model_dir models/dfm_large --output report_test.json
```

The scripts use GPU automatically if available. Adjust training hyperparameters in `train.py` or via command-line args.
