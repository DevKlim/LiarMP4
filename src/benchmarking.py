import pandas as pd
import numpy as np
import shutil
import json
import math
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Lazy import to avoid startup overhead
try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False

DATA_AI = Path("data/dataset.csv")
DATA_MANUAL = Path("data/manual_dataset.csv")

def sanitize_for_json(obj):
    """Recursively clean floats for JSON output."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj): return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj

def get_combined_dataset():
    """
    Joins AI predictions with Manual Ground Truth on ID.
    """
    if not DATA_AI.exists() or not DATA_MANUAL.exists():
        return None

    try:
        # Load datasets
        df_ai = pd.read_csv(DATA_AI)
        df_manual = pd.read_csv(DATA_MANUAL)
        
        # Normalize IDs (Trim spaces, ensure string)
        df_ai['id'] = df_ai['id'].astype(str).str.strip()
        df_manual['id'] = df_manual['id'].astype(str).str.strip()

        # Force numeric conversion for scores
        df_ai['final_veracity_score'] = pd.to_numeric(df_ai['final_veracity_score'], errors='coerce').fillna(0)
        df_manual['final_veracity_score'] = pd.to_numeric(df_manual['final_veracity_score'], errors='coerce').fillna(0)
        
        # Merge on ID
        merged = pd.merge(
            df_ai,
            df_manual[['id', 'final_veracity_score', 'classification']], 
            on='id', 
            suffixes=('_ai', '_manual'),
            how='inner'
        )
        
        # Calculate Errors
        merged['score_diff'] = merged['final_veracity_score_ai'] - merged['final_veracity_score_manual']
        merged['abs_error'] = merged['score_diff'].abs()
        
        return merged
    except Exception as e:
        print(f"Error merging datasets: {e}")
        return None

def format_config_params(params_raw):
    """Parses the config_params JSON string into a readable format for the leaderboard."""
    if pd.isna(params_raw) or not params_raw:
        return "Defaults"
    try:
        if isinstance(params_raw, str):
            p = json.loads(params_raw)
        else:
            p = params_raw
            
        # Extract key differentiators
        reprompts = p.get('reprompts', 0)
        comments = "Yes" if p.get('include_comments') == 'true' or p.get('include_comments') is True else "No"
        return f"Retries:{reprompts} | Context:{comments}"
    except:
        return "Legacy/Unknown"

def calculate_benchmarks():
    """Global stats (All AI models vs Ground Truth)."""
    merged = get_combined_dataset()
    if merged is None or len(merged) == 0:
        return {"status": "no_data"}
    
    mae = merged['abs_error'].mean()
    
    # Binary Accuracy (Threshold 50)
    merged['bin_ai'] = merged['final_veracity_score_ai'] >= 50
    merged['bin_manual'] = merged['final_veracity_score_manual'] >= 50
    accuracy = (merged['bin_ai'] == merged['bin_manual']).mean()
    
    recent_samples = merged.tail(5)[['id', 'abs_error', 'final_veracity_score_ai', 'final_veracity_score_manual']].to_dict(orient='records')

    result = {
        "count": int(len(merged)),
        "mae": round(mae, 2),
        "accuracy_percent": round(accuracy * 100, 1),
        "recent_samples": recent_samples
    }
    return sanitize_for_json(result)

def generate_leaderboard():
    """
    Groups results by Configuration to rank models/prompts.
    """
    merged = get_combined_dataset()
    if merged is None or len(merged) == 0:
        return []

    # Fill missing config columns for legacy data
    for col in ['config_model', 'config_prompt', 'config_reasoning', 'config_params']:
        if col not in merged.columns: merged[col] = "Unknown"
        
    merged = merged.fillna({'config_model': 'Unknown', 'config_prompt': 'Standard', 'config_reasoning': 'None'})

    # Make params readable
    merged['params_readable'] = merged['config_params'].apply(format_config_params)

    # Calculate Correctness
    merged['bin_ai'] = merged['final_veracity_score_ai'] >= 50
    merged['bin_manual'] = merged['final_veracity_score_manual'] >= 50
    merged['is_correct'] = (merged['bin_ai'] == merged['bin_manual']).astype(int)

    # Group By Configuration
    grouped = merged.groupby(['config_model', 'config_prompt', 'config_reasoning', 'params_readable']).agg(
        mae=('abs_error', 'mean'),
        accuracy=('is_correct', 'mean'),
        count=('id', 'count')
    ).reset_index()

    # Format Output
    leaderboard = []
    for _, row in grouped.iterrows():
        leaderboard.append({
            "type": "GenAI",
            "model": row['config_model'],
            "prompt": row['config_prompt'],
            "reasoning": row['config_reasoning'],
            "params": row['params_readable'],
            "mae": round(row['mae'], 2),
            "accuracy": round(row['accuracy'] * 100, 1),
            "samples": int(row['count'])
        })

    # Sort: Highest Accuracy, then Lowest MAE
    leaderboard.sort(key=lambda x: (-x['accuracy'], x['mae']))
    
    return sanitize_for_json(leaderboard)

def train_predictive_sandbox(features_config: dict):
    """
    Trains simple models on Ground Truth to set a baseline.
    """
    if not DATA_MANUAL.exists(): return {"error": "No data"}
    df = pd.read_csv(DATA_MANUAL).dropna(subset=['caption', 'final_veracity_score'])
    if len(df) < 5: return {"error": "Not enough data"}

    # Features
    df['len'] = df['caption'].astype(str).apply(len)
    keywords = ["shocking", "breaking", "watch"]
    df['kw_count'] = df['caption'].astype(str).apply(lambda x: sum(1 for k in keywords if k in x.lower()))
    feat_cols = ['len', 'kw_count']
    
    # Target
    df['target'] = (pd.to_numeric(df['final_veracity_score'], errors='coerce').fillna(0) >= 50).astype(int)

    # Simple Logistic Regression Baseline
    try:
        X_train, X_test, y_train, y_test = train_test_split(df[feat_cols], df['target'], test_size=0.3, random_state=42)
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        return {
            "status": "success",
            "type": "logistic_regression",
            "accuracy": round(clf.score(X_test, y_test) * 100, 1),
            "message": "Baseline trained on Caption Length + Keywords."
        }
    except Exception as e:
        return {"error": str(e)}
