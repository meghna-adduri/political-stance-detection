# Stance Detection

Political stance detection on tweets, using the P-Stance dataset.

## Approach

A single model is trained across all three targets (Trump, Biden, Bernie)
rather than a separate model per target. The target is passed as part of the
input alongside the tweet — see `format_prompt` in `src/data.py`, which
builds a prompt of the form "Tweet: ... Target: ... Question: Does the
author of this tweet favor or oppose {target}? Answer:" — and the model
predicts stance conditioned on both. Training data from all three targets is
pooled (`src/data.py` combines the per-target CSVs into
`data/processed/{split}_combined.csv`). A unified model also allows
cross-target evaluation, i.e. checking performance on a target using
examples the model wasn't trained on for that target.

## Setup

Clone the repo and set up a virtual environment:

```bash
git clone <repo-url>
cd political-stance-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Core dependencies: pandas, transformers, peft, accelerate, wandb, pytest.

### Data

Dataset: [P-Stance](https://github.com/chuchun8/PStance) (chuchun8/PStance),
MIT licensed. The repo itself only holds code; the labeled CSVs are hosted
separately on Google Drive (linked from the P-Stance repo). Download them and
place them in `data/raw/`.

`data/raw/` contains 9 CSVs: train/val/test splits for each of three targets
(Trump, Biden, Bernie). Each file has three columns: `Tweet`, `Target`,
`Stance` (`FAVOR` or `AGAINST`). Tweet text is included directly in the CSVs,
so no ID hydration against the Twitter API is needed. Roughly 21,574 labeled
tweets total across all files.

`data/processed/` holds derived files built from `data/raw/`, e.g. the
combined training set described below.
