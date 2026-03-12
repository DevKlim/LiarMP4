import pandas as pd
import numpy as np
import shutil
import json
import math
from pathlib import Path

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
        return[sanitize_for_json(v) for v in obj]
    return obj

def calculate_tag_accuracy(tags_ai, tags_man):
    if pd.isna(tags_ai): tags_ai = ""
    if pd.isna(tags_man): tags_man = ""
    set_ai = set([t.strip().lower() for t in str(tags_ai).split(',') if t.strip()])
    set_man = set([t.strip().lower() for t in str(tags_man).split(',') if t.strip()])
    if not set_man and not set_ai: return 1.0
    if not set_man or not set_ai: return 0.0
    # Jaccard Similarity
    return len(set_ai.intersection(set_man)) / len(set_ai.union(set_man))

def get_combined_dataset():
    """
    Joins AI predictions with Manual Ground Truth on ID and calculates comprehensive vector differences.
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

        df_manual_cols =['id', 'final_veracity_score', 'visual_integrity_score', 'audio_integrity_score', 'source_credibility_score', 'logical_consistency_score', 'emotional_manipulation_score', 'video_audio_score', 'video_caption_score', 'audio_caption_score', 'tags', 'classification']
        
        # Merge on ID
        merged = pd.merge(
            df_ai,
            df_manual[[c for c in df_manual_cols if c in df_manual.columns]], 
            on='id', 
            suffixes=('_ai', '_manual'),
            how='inner'
        )
        
        # 1. Final Score Error
        merged['final_veracity_score_ai'] = pd.to_numeric(merged['final_veracity_score_ai'], errors='coerce').fillna(0)
        merged['final_veracity_score_manual'] = pd.to_numeric(merged['final_veracity_score_manual'], errors='coerce').fillna(0)
        merged['abs_error'] = (merged['final_veracity_score_ai'] - merged['final_veracity_score_manual']).abs()
        
        # 2. Sophisticated Vector Calculations
        vector_pairs =[
            ('visual_score', 'visual_integrity_score'),
            ('audio_score', 'audio_integrity_score'),
            ('source_score', 'source_credibility_score'),
            ('logic_score', 'logical_consistency_score'),
            ('emotion_score', 'emotional_manipulation_score'),
            ('align_video_audio', 'video_audio_score'),
            ('align_video_caption', 'video_caption_score'),
            ('align_audio_caption', 'audio_caption_score'),
        ]
        
        error_cols = ['abs_error']
        for ai_c, man_c in vector_pairs:
            if ai_c in merged.columns and man_c in merged.columns:
                # Multiply 1-10 scores by 10 to put them on the same 0-100 scale as final score
                merged[ai_c] = pd.to_numeric(merged[ai_c], errors='coerce').fillna(5) * 10
                merged[man_c] = pd.to_numeric(merged[man_c], errors='coerce').fillna(5) * 10
                err_c = f"err_{ai_c}"
                merged[err_c] = (merged[ai_c] - merged[man_c]).abs()
                error_cols.append(err_c)

        # Composite MAE represents the mean absolute error across the final score AND all 8 sub-vectors
        merged['composite_mae'] = merged[error_cols].mean(axis=1)
        
        # 3. Tag Accuracy Calculation
        merged['tag_accuracy'] = merged.apply(lambda row: calculate_tag_accuracy(row.get('tags_ai', ''), row.get('tags_manual', '')), axis=1)
        
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
    
    mae = merged['composite_mae'].mean()
    tag_acc = merged['tag_accuracy'].mean()
    
    # Binary Accuracy (Threshold 50)
    merged['bin_ai'] = merged['final_veracity_score_ai'] >= 50
    merged['bin_manual'] = merged['final_veracity_score_manual'] >= 50
    accuracy = (merged['bin_ai'] == merged['bin_manual']).mean()
    
    recent_samples = merged.tail(5)[['id', 'composite_mae', 'final_veracity_score_ai', 'final_veracity_score_manual']].to_dict(orient='records')

    result = {
        "count": int(len(merged)),
        "mae": round(mae, 2), # Exposing composite MAE as main MAE metric
        "accuracy_percent": round(accuracy * 100, 1),
        "tag_accuracy_percent": round(tag_acc * 100, 1),
        "recent_samples": recent_samples
    }
    return sanitize_for_json(result)

def generate_leaderboard():
    """
    Groups results by Configuration to rank models/prompts using sophisticated distance measurements.
    """
    merged = get_combined_dataset()
    if merged is None or len(merged) == 0:
        return []

    for col in['config_model', 'config_prompt', 'config_reasoning', 'config_params']:
        if col not in merged.columns: merged[col] = "Unknown"
        
    merged = merged.fillna({'config_model': 'Unknown', 'config_prompt': 'Standard', 'config_reasoning': 'None'})

    merged['params_readable'] = merged['config_params'].apply(format_config_params)
    
    def extract_tools(p_raw):
        try:
            if isinstance(p_raw, str): p = json.loads(p_raw)
            else: p = p_raw
            if not isinstance(p, dict): return "None"
            tools =[]
            if p.get('agent_active'): tools.append("Agent")
            if p.get('use_search'): tools.append("Search")
            if p.get('use_code'): tools.append("Code")
            if p.get('few_shot') or p.get('multi_shot'): tools.append("Few-Shot")
            return ", ".join(tools) if tools else "None"
        except:
            return "None"

    merged['tools'] = merged['config_params'].apply(extract_tools)

    merged['bin_ai'] = merged['final_veracity_score_ai'] >= 50
    merged['bin_manual'] = merged['final_veracity_score_manual'] >= 50
    merged['is_correct'] = (merged['bin_ai'] == merged['bin_manual']).astype(int)

    def get_fcot_depth(row):
        r = str(row['config_reasoning']).lower()
        if 'fcot' in r: return 2
        elif 'cot' in r: return 1
        return 0
    merged['fcot_depth'] = merged.apply(get_fcot_depth, axis=1)

    agg_dict = {
        'comp_mae': ('composite_mae', 'mean'),
        'tag_accuracy': ('tag_accuracy', 'mean'),
        'accuracy': ('is_correct', 'mean'),
        'count': ('id', 'count')
    }
    
    err_cols =[
        'err_visual_score', 'err_audio_score', 'err_source_score',
        'err_logic_score', 'err_emotion_score', 'err_align_video_audio',
        'err_align_video_caption', 'err_align_audio_caption'
    ]
    for col in err_cols:
        if col in merged.columns:
            agg_dict[col] = (col, 'mean')

    # Group By Configuration using Composite MAE and Tag Accuracy
    grouped = merged.groupby(['config_model', 'config_prompt', 'config_reasoning', 'params_readable', 'tools', 'fcot_depth']).agg(**agg_dict).reset_index()

    leaderboard =[]
    for _, row in grouped.iterrows():
        entry = {
            "type": "GenAI",
            "model": row['config_model'],
            "prompt": row['config_prompt'],
            "reasoning": row['config_reasoning'],
            "params": row['params_readable'],
            "tools": row['tools'],
            "fcot_depth": int(row['fcot_depth']),
            "comp_mae": round(row['comp_mae'], 2),
            "tag_acc": round(row['tag_accuracy'] * 100, 1),
            "accuracy": round(row['accuracy'] * 100, 1),
            "samples": int(row['count'])
        }
        for col in err_cols:
            if col in row:
                entry[col] = round(row[col], 2)
        leaderboard.append(entry)

    # Sort: Highest Accuracy, Highest Tag Accuracy, then Lowest Composite MAE
    leaderboard.sort(key=lambda x: (-x['accuracy'], -x['tag_acc'], x['comp_mae']))
    
    return sanitize_for_json(leaderboard)
