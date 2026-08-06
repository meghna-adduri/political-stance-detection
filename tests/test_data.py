import pandas as pd
from src.data import load_combined, format_prompt, TARGETS

def test_combined_row_count_matches_sum_of_targets():
    combined = load_combined('train')
    individual_total = sum(
        len(pd.read_csv(f"data/raw/raw_train_{t}.csv")) for t in TARGETS
    )
    assert len(combined) == individual_total

def test_labels_are_expected_set():
    combined = load_combined('train')
    assert set(combined['Stance'].unique()) <= {'FAVOR', 'AGAINST'}

def test_every_row_has_target():
    combined = load_combined('train')
    assert combined['Target'].notna().all()

def test_format_prompt_includes_inputs():
    prompt = format_prompt("some tweet text", "trump")
    assert "some tweet text" in prompt
    assert "trump" in prompt