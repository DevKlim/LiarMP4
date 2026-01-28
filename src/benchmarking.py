import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import re

# Lazy import to avoid startup overhead if not used immediately
try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False

DATA_AI = Path("data/dataset.csv")
DATA_MANUAL = Path("data/manual_dataset.csv")

def get_combined_dataset():
    """
    Joins AI predictions with Manual Ground Truth on ID.
    """
    if not DATA_AI.exists() or not DATA_MANUAL.exists():
        return None

    try:
        df_ai = pd.read_csv(DATA_AI)
        df_manual = pd.read_csv(DATA_MANUAL)
        
        # Ensure IDs are strings
        df_ai['id'] = df_ai['id'].astype(str)
        df_manual['id'] = df_manual['id'].astype(str)
        
        # Merge on ID
        merged = pd.merge(
            df_ai[['id', 'final_veracity_score', 'timestamp', 'classification']], 
            df_manual[['id', 'final_veracity_score', 'classification', 'caption']], 
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

def calculate_benchmarks():
    """
    Returns statistical comparison between AI and Ground Truth.
    """
    merged = get_combined_dataset()
    if merged is None or len(merged) == 0:
        return {"status": "no_data"}
    
    mae = merged['abs_error'].mean()
    mse = (merged['abs_error'] ** 2).mean()
    
    # Categorize Agreement (Binary: Fake < 50, Real >= 50)
    merged['bin_ai'] = merged['final_veracity_score_ai'] >= 50
    merged['bin_manual'] = merged['final_veracity_score_manual'] >= 50
    accuracy = (merged['bin_ai'] == merged['bin_manual']).mean()
    
    return {
        "count": int(len(merged)),
        "mae": round(mae, 2),
        "accuracy_percent": round(accuracy * 100, 1),
        "recent_samples": merged.tail(5).to_dict(orient='records')
    }

def train_with_autogluon(df, feature_cols, target_col):
    if not AUTOGLUON_AVAILABLE:
        return {"error": "AutoGluon library not installed."}

    # Setup Sandbox Directory
    sandbox_path = "models/sandbox_autogluon"
    if Path(sandbox_path).exists():
        shutil.rmtree(sandbox_path)
    
    # Prepare Data
    train_data, test_data = train_test_split(df[feature_cols + [target_col]], test_size=0.3, random_state=42)
    
    try:
        # Train
        predictor = TabularPredictor(label=target_col, path=sandbox_path, verbosity=2).fit(
            train_data=train_data,
            time_limit=45, # Fast sandbox
            presets='medium_quality' # XGBoost, CatBoost, LightGBM, RF
        )
        
        # Leaderboard
        lb = predictor.leaderboard(test_data, silent=True)
        # Convert to simple list of dicts
        results = lb[['model', 'score_test', 'score_val', 'fit_time']].to_dict(orient='records')
        
        return {
            "status": "success",
            "type": "autogluon",
            "leaderboard": results,
            "best_model": predictor.get_model_best(),
            "message": "AutoML training complete. Evaluated multiple gradient boosting frameworks."
        }
    except Exception as e:
        return {"error": f"AutoGluon training failed: {str(e)}"}

def train_predictive_sandbox(features_config: dict):
    """
    Trains models on the Manual Dataset (Ground Truth).
    Supports 'logistic' (sklearn) and 'autogluon' (AutoML).
    """
    if not DATA_MANUAL.exists():
        return {"error": "No ground truth data found."}
        
    df = pd.read_csv(DATA_MANUAL)
    # Ensure mandatory fields
    if 'caption' not in df.columns or 'final_veracity_score' not in df.columns:
        return {"error": "Dataset missing 'caption' or 'final_veracity_score' columns."}

    df = df.dropna(subset=['caption', 'final_veracity_score'])
    
    if len(df) < 5:
        return {"error": "Need at least 5 manual labels to train."}

    # --- Feature Engineering ---
    # 1. Caption Length
    df['feat_len'] = df['caption'].astype(str).apply(len)
    
    # 2. Keyword Flags (Simple Heuristic)
    keywords = ["shocking", "breaking", "urgent", "watch", "leaked", "official"]
    df['feat_keywords'] = df['caption'].astype(str).apply(lambda x: sum(1 for k in keywords if k in x.lower()))
    
    # 3. New granular features if available in Ground Truth
    feature_cols = ['feat_len', 'feat_keywords']
    
    # Check for the requested "visual_integrity_score" etc.
    # If using them as input features, it implies we are training a meta-model or simulating their availability.
    # Typically, these are TARGETS or Outputs of GenAI, but for a predictive model, 
    # we usually only have metadata. 
    # However, if 'use_visual_meta' is true, we assume we have some visual score available (perhaps from metadata API).
    use_visual = features_config.get('use_visual_meta', False)
    
    if use_visual:
        # Try to use specific columns if they exist, fallback to generic 'visual_score'
        if 'visual_integrity_score' in df.columns:
             # Clean and convert to float
             df['visual_integrity_score'] = pd.to_numeric(df['visual_integrity_score'], errors='coerce').fillna(0)
             feature_cols.append('visual_integrity_score')
        elif 'visual_score' in df.columns:
             df['visual_score'] = pd.to_numeric(df['visual_score'], errors='coerce').fillna(0)
             feature_cols.append('visual_score')

    # Target: High Veracity (>=50) vs Low
    target_col = 'target_veracity'
    df[target_col] = (df['final_veracity_score'].astype(float) >= 50).astype(int)

    model_type = features_config.get('model_type', 'logistic')

    # --- AutoGluon Branch ---
    if model_type == 'autogluon':
        return train_with_autogluon(df, feature_cols, target_col)

    # --- Logistic Branch ---
    X = df[feature_cols]
    y = df[target_col]
    
    try:
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train Model
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        
        # Evaluate
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        
        # Coefficients interpretation
        coefs = dict(zip(feature_cols, clf.coef_[0].tolist()))
        
        return {
            "status": "success",
            "type": "logistic",
            "samples": len(df),
            "train_accuracy": round(train_acc * 100, 1),
            "test_accuracy": round(test_acc * 100, 1),
            "feature_importance": coefs,
            "message": "Model trained on available features."
        }
    except Exception as e:
        return {"error": str(e)}
