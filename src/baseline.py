import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import wandb
import re

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

def get_prediction(model, tokenizer, tweet: str, target: str) -> str:
    prompt = format_prompt(tweet, target)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=10,   # short, you just need FAVOR/AGAINST
            do_sample=False       # greedy decoding, deterministic and reproducible
        )
    response = tokenizer.decode(
        output_ids[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    return parse_label(response)

def parse_label(response: str) -> str:
    """Map raw model output to FAVOR/AGAINST, handling variation in phrasing."""
    response_upper = response.upper()
    if "FAVOR" in response_upper or "SUPPORT" in response_upper:
        return "FAVOR"
    elif "AGAINST" in response_upper or "OPPOSE" in response_upper:
        return "AGAINST"
    else:
        return "UNKNOWN"

def run_baseline():
    model, tokenizer = load_model()
    test_df = load_combined('test')

    predictions = []
    for _, row in test_df.iterrows():
        pred = get_prediction(model, tokenizer, row['Tweet'], row['Target'])
        predictions.append(pred)

    test_df['prediction'] = predictions
    return test_df