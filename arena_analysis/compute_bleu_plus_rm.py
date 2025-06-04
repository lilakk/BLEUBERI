import pandas as pd
from scipy.stats import zscore
import numpy as np

csv_path = "arena_1k_final_results/arena_benchmark_results_claude.csv"
df = pd.read_csv(csv_path)

cols_to_standardize = ['bleu_score_a', 'bleu_score_b', 'rm_8b_score_a', 'rm_8b_score_b', 'rm_27b_score_a', 'rm_27b_score_b']

for col in cols_to_standardize:
    if col in df.columns:
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        
        mean_val = numeric_series.mean()
        std_val = numeric_series.std()
        
        if std_val == 0 or pd.isna(std_val):
            df[f'{col}_scaled'] = 0.0 if numeric_series.notna().any() else np.nan
        else:
            df[f'{col}_scaled'] = (numeric_series - mean_val) / std_val
    else:
        print(f"Warning: Column '{col}' not found. Skipping z-score standardization for it.")
        df[f'{col}_scaled'] = np.nan

df['bleu_rm8b_scaled_a'] = (df.get('bleu_score_a_scaled', np.nan) + df.get('rm_8b_score_a_scaled', np.nan)) / 2
df['bleu_rm8b_scaled_b'] = (df.get('bleu_score_b_scaled', np.nan) + df.get('rm_8b_score_b_scaled', np.nan)) / 2
df['bleu_rm27b_scaled_a'] = (df.get('bleu_score_a_scaled', np.nan) + df.get('rm_27b_score_a_scaled', np.nan)) / 2
df['bleu_rm27b_scaled_b'] = (df.get('bleu_score_b_scaled', np.nan) + df.get('rm_27b_score_b_scaled', np.nan)) / 2

df['bleu_rm8b_winner'] = 'tie'

valid_comparison_mask = df['bleu_rm8b_scaled_a'].notna() & df['bleu_rm8b_scaled_b'].notna()

df.loc[valid_comparison_mask & (df['bleu_rm8b_scaled_a'] > df['bleu_rm8b_scaled_b']), 'bleu_rm8b_winner'] = 'model_a'
df.loc[valid_comparison_mask & (df['bleu_rm8b_scaled_b'] > df['bleu_rm8b_scaled_a']), 'bleu_rm8b_winner'] = 'model_b'

df['bleu_rm27b_winner'] = 'tie'
valid_comparison_mask_rm27b = df['bleu_rm27b_scaled_a'].notna() & df['bleu_rm27b_scaled_b'].notna()
df.loc[valid_comparison_mask_rm27b & (df['bleu_rm27b_scaled_a'] > df['bleu_rm27b_scaled_b']), 'bleu_rm27b_winner'] = 'model_a'
df.loc[valid_comparison_mask_rm27b & (df['bleu_rm27b_scaled_b'] > df['bleu_rm27b_scaled_a']), 'bleu_rm27b_winner'] = 'model_b'

human_winner_col = 'winner'

if human_winner_col in df.columns:
    tie_normalization_map = {
        'tie (bothbad)': 'tie',
        'tie (ambiguous)': 'tie',
        'tie (both good)': 'tie',
        'tie (close)': 'tie',
        'tie - both bad': 'tie',
        'tie - both good': 'tie',
    }
    df['winner_for_comparison'] = df[human_winner_col].astype(str).str.lower()
    df['winner_for_comparison'] = df['winner_for_comparison'].replace(tie_normalization_map)
    
    df.loc[df['winner_for_comparison'].str.contains('model_a', case=False, na=False), 'winner_for_comparison'] = 'model_a'
    df.loc[df['winner_for_comparison'].str.contains('model_b', case=False, na=False), 'winner_for_comparison'] = 'model_b'
    df.loc[df['winner_for_comparison'].str.contains('tie', case=False, na=False), 'winner_for_comparison'] = 'tie'

    print(f"\n--- Agreement of Original Metrics (Unscaled) with Human Preference ---")

    if 'bleu_winner' in df.columns:
        df['bleu_winner_for_comparison'] = df['bleu_winner'].astype(str).str.lower()
        df['bleu_winner_for_comparison'] = df['bleu_winner_for_comparison'].replace(tie_normalization_map)
        df.loc[df['bleu_winner_for_comparison'].str.contains('model_a', case=False, na=False), 'bleu_winner_for_comparison'] = 'model_a'
        df.loc[df['bleu_winner_for_comparison'].str.contains('model_b', case=False, na=False), 'bleu_winner_for_comparison'] = 'model_b'
        df.loc[df['bleu_winner_for_comparison'].str.contains('tie', case=False, na=False), 'bleu_winner_for_comparison'] = 'tie'
        
        df['original_bleu_alignment'] = (df['bleu_winner_for_comparison'] == df['winner_for_comparison'])
        original_bleu_agreement_rate = df['original_bleu_alignment'].mean()
        print(f"Agreement between original 'bleu_winner' and human preference: {original_bleu_agreement_rate:.4f}")
        
        print(f"Normalized 'bleu_winner_for_comparison' value counts:")
        print(df['bleu_winner_for_comparison'].value_counts(dropna=False))
    else:
        print("Warning: 'bleu_winner' column not found. Cannot calculate its agreement.")

    if 'rm_8b_winner' in df.columns:
        df['rm_8b_winner_for_comparison'] = df['rm_8b_winner'].astype(str).str.lower()
        df['rm_8b_winner_for_comparison'] = df['rm_8b_winner_for_comparison'].replace(tie_normalization_map)
        df.loc[df['rm_8b_winner_for_comparison'].str.contains('model_a', case=False, na=False), 'rm_8b_winner_for_comparison'] = 'model_a'
        df.loc[df['rm_8b_winner_for_comparison'].str.contains('model_b', case=False, na=False), 'rm_8b_winner_for_comparison'] = 'model_b'
        df.loc[df['rm_8b_winner_for_comparison'].str.contains('tie', case=False, na=False), 'rm_8b_winner_for_comparison'] = 'tie'

        df['original_rm_8b_alignment'] = (df['rm_8b_winner_for_comparison'] == df['winner_for_comparison'])
        original_rm_8b_agreement_rate = df['original_rm_8b_alignment'].mean()
        print(f"Agreement between original 'rm_8b_winner' and human preference: {original_rm_8b_agreement_rate:.4f}")

        print(f"Normalized 'rm_8b_winner_for_comparison' value counts:")
        print(df['rm_8b_winner_for_comparison'].value_counts(dropna=False))
    else:
        print("Warning: 'rm_8b_winner' column not found. Cannot calculate its agreement.")

    if 'rm_27b_winner' in df.columns:
        df['rm_27b_winner_for_comparison'] = df['rm_27b_winner'].astype(str).str.lower()
        df['rm_27b_winner_for_comparison'] = df['rm_27b_winner_for_comparison'].replace(tie_normalization_map)
        df.loc[df['rm_27b_winner_for_comparison'].str.contains('model_a', case=False, na=False), 'rm_27b_winner_for_comparison'] = 'model_a'
        df.loc[df['rm_27b_winner_for_comparison'].str.contains('model_b', case=False, na=False), 'rm_27b_winner_for_comparison'] = 'model_b'
        df.loc[df['rm_27b_winner_for_comparison'].str.contains('tie', case=False, na=False), 'rm_27b_winner_for_comparison'] = 'tie'

        df['original_rm_27b_alignment'] = (df['rm_27b_winner_for_comparison'] == df['winner_for_comparison'])
        original_rm_27b_agreement_rate = df['original_rm_27b_alignment'].mean()
        print(f"Agreement between original 'rm_27b_winner' and human preference: {original_rm_27b_agreement_rate:.4f}")

        print(f"Normalized 'rm_27b_winner_for_comparison' value counts:")
        print(df['rm_27b_winner_for_comparison'].value_counts(dropna=False))
    else:
        print("Warning: 'rm_27b_winner' column not found. Cannot calculate its agreement.")

    df['bleu_rm8b_alignment'] = (df['bleu_rm8b_winner'] == df['winner_for_comparison'])
    agreement_rate = df['bleu_rm8b_alignment'].mean()

    df['bleu_rm27b_alignment'] = (df['bleu_rm27b_winner'] == df['winner_for_comparison'])
    agreement_rate_bleu_rm27b = df['bleu_rm27b_alignment'].mean()

    print(f"--- DataFrame Head after Processing (sample of relevant columns) ---")
    cols_to_print = ['id', human_winner_col, 'winner_for_comparison',
                     'bleu_score_a_scaled', 'rm_8b_score_a_scaled', 'rm_27b_score_a_scaled',
                     'bleu_rm8b_scaled_a', 'bleu_rm27b_scaled_a',
                     'bleu_score_b_scaled', 'rm_8b_score_b_scaled', 'rm_27b_score_b_scaled',
                     'bleu_rm8b_scaled_b', 'bleu_rm27b_scaled_b',
                     'bleu_rm8b_winner', 'bleu_rm27b_winner',
                     'bleu_rm8b_alignment', 'bleu_rm27b_alignment']
    cols_to_print = [col for col in cols_to_print if col in df.columns]
    print(df[cols_to_print].head(10))

    print(f"\n--- Value Counts for Winner Columns ---")
    print(f"Original human preference ('{human_winner_col}') value counts (Top 5 unique):")
    print(df[human_winner_col].value_counts(dropna=False).nlargest(5))
    if df[human_winner_col].nunique(dropna=False) > 5:
        print(f"...and {df[human_winner_col].nunique(dropna=False) - 5} more unique value(s).")

    print(f"Normalized human preference ('winner_for_comparison') value counts:")
    print(df['winner_for_comparison'].value_counts(dropna=False))
    
    print(f"Predicted 'bleu_rm8b_winner' value counts:")
    print(df['bleu_rm8b_winner'].value_counts(dropna=False))

    print(f"Predicted 'bleu_rm27b_winner' value counts:")
    print(df['bleu_rm27b_winner'].value_counts(dropna=False))

    print(f"\nAgreement between 'bleu-rm8b' combined metric and human preference: {agreement_rate:.4f}")
    print(f"Agreement between 'bleu-rm27b' combined metric and human preference: {agreement_rate_bleu_rm27b:.4f}")
else:
    print(f"Error: Human winner column '{human_winner_col}' not found in DataFrame.")
    print("Cannot compute agreement. Please check the CSV file and column names.")
