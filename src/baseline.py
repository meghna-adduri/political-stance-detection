import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, f1_score
import wandb
from tqdm import tqdm

from src.data import load_combined, format_prompt

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

def load_model():
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=quantization_config
    )
    return model, tokenizer

def get_prediction(model, tokenizer, tweet: str, target: str) -> tuple[str, str]:
    prompt = format_prompt(tweet, target)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )
    response = tokenizer.decode(
        output_ids[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    return response, parse_label(response)

def parse_label(response: str) -> str:
    """Map raw model output to FAVOR/AGAINST, handling variation in phrasing."""
    response_upper = response.upper()
    if "FAVOR" in response_upper or "SUPPORT" in response_upper:
        return "FAVOR"
    elif "AGAINST" in response_upper or "OPPOSE" in response_upper:
        return "AGAINST"
    else:
        return "UNKNOWN"

def run_baseline(model, tokenizer):
    test_df = load_combined('test')

    raw_responses = []
    predictions = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        raw, pred = get_prediction(model, tokenizer, row['Tweet'], row['Target'])
        raw_responses.append(raw)
        predictions.append(pred)

    test_df['raw_response'] = raw_responses
    test_df['prediction'] = predictions

    os.makedirs("data/processed", exist_ok=True)
    test_df.to_csv("data/processed/zeroshot_predictions.csv", index=False)

    return test_df

def evaluate_and_log(test_df):
    valid = test_df[test_df['prediction'] != 'UNKNOWN']
    dropped = len(test_df) - len(valid)

    acc = accuracy_score(valid['Stance'], valid['prediction'])
    macro_f1 = f1_score(valid['Stance'], valid['prediction'], average='macro')

    wandb.init(project="political-stance-detection", name="zeroshot-qwen2.5-7b")
    wandb.log({
        "accuracy": acc,
        "macro_f1": macro_f1,
        "unknown_predictions": dropped,
        "total_examples": len(test_df)
    })

    for target in test_df['Target'].unique():
        subset = valid[valid['Target'] == target]
        wandb.log({
            f"accuracy_{target}": accuracy_score(subset['Stance'], subset['prediction']),
            f"macro_f1_{target}": f1_score(subset['Stance'], subset['prediction'], average='macro')
        })

    wandb.finish()
    return acc, macro_f1