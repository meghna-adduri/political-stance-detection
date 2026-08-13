import pandas as pd

TARGETS = ['trump', 'biden', 'bernie']

def load_combined(split: str) -> pd.DataFrame:
    dfs = []
    for target in TARGETS:
        df = pd.read_csv(f"data/raw/raw_{split}_{target}.csv")
        if 'Target' not in df.columns:
            df['Target'] = target
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def format_prompt(tweet: str, target: str) -> str:
    return (
        f"Tweet: {tweet}\n"
        f"Target: {target}\n"
        f"Does the author of this tweet favor or oppose {target}? "
        f"Respond with only the single word FAVOR or AGAINST. "
        f"Do not begin your answer with any preamble like 'Based on the content of the tweet' or "
        f"'Looking at the tweet.' Do not explain your reasoning.\n"
        f"Answer:"
    )

if __name__ == "__main__":
    for split in ['train', 'val', 'test']:
        df = load_combined(split)
        df.to_csv(f"data/processed/{split}_combined.csv", index=False)