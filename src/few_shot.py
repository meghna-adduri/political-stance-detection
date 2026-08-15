import os
import json
from tqdm import tqdm

from src.data import load_combined, format_few_shot_prompt
from src.baseline import get_prediction

def load_few_shot_examples(path="data/processed/few_shot_examples.json"):
    with open(path) as f:
        return json.load(f)

def run_few_shot(model, tokenizer, examples):
    test_df = load_combined('test')

    raw_responses = []
    predictions = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        prompt = format_few_shot_prompt(row['Tweet'], row['Target'], examples)
        raw, pred = get_prediction(model, tokenizer, prompt)
        raw_responses.append(raw)
        predictions.append(pred)

    test_df['raw_response'] = raw_responses
    test_df['prediction'] = predictions

    os.makedirs("data/processed", exist_ok=True)
    test_df.to_csv("data/processed/fewshot_predictions.csv", index=False)

    return test_df