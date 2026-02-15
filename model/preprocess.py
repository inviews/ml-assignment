

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Paths

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_PATH = os.path.join(DATA_DIR, "online_shoppers_intention.csv")
CLEAN_PATH = os.path.join(DATA_DIR, "online_shoppers_intention_clean.csv")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")

RANDOM_STATE = 42
TEST_SIZE = 0.20


#  Load

def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    print(f"[load]  Raw shape: {df.shape}")
    return df




EXPECTED_COLUMNS = [
    "Administrative", "Administrative_Duration", "Informational", "Informational_Duration", 
    "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", 
    "SpecialDay", "Month", "OperatingSystems", "Browser", "Region", "TrafficType", 
    "VisitorType", "Weekend", "Revenue"
]

def validate_schema(df: pd.DataFrame) -> None:
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    print("[validate] Schema OK — all columns present.")




def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    n_missing = df.isnull().sum().sum()
    if n_missing == 0:
        print("[missing] No missing values found.")
        return df

    print(f"[missing] {n_missing} missing values detected.")

    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object' or df[col].dtype == 'bool':
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
            else:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                
    return df




def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    print(f"[dedup]  Removed {n_removed} duplicate rows ({n_removed/n_before:.1%}). "
          f"Remaining: {len(df)}")
    return df




def encode_features(df: pd.DataFrame) -> pd.DataFrame:

    df['Weekend'] = df['Weekend'].astype(int)
    df['Revenue'] = df['Revenue'].astype(int)
    
    categorical_cols = ['Month', 'VisitorType', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']
    
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True) 

    
    print(f"[encode] One-Hot Encoded {categorical_cols}. New shape: {df.shape}")
    return df




def report_correlations(df: pd.DataFrame, target_col: str = 'Revenue', top_n: int = 10) -> None:
    if target_col not in df.columns:
        return
    corr = df.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
    print(f"\n[corr]   Top {top_n} features by |correlation| with target:")
    for rank, (feat, val) in enumerate(corr.head(top_n).items(), 1):
        print(f"           {rank:2d}. {feat:<35s}  {val:.4f}")
    print()



def split_and_save(df: pd.DataFrame) -> dict:
    target_col = 'Revenue'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"[split]  Train: {len(X_train)}  |  Test: {len(X_test)}")

    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Save cleaned full dataset
    df.to_csv(CLEAN_PATH, index=False)
    print(f"[save]   Clean dataset → {CLEAN_PATH}")

    # Save splits
    for name, data in [("X_train", X_train), ("X_test", X_test),
                       ("y_train", y_train), ("y_test", y_test)]:
        out_path = os.path.join(DATA_DIR, f"{name}.csv")
        data.to_csv(out_path, index=False)
        print(f"[save]   {name:<10s} → {out_path}")

    # Save scaler
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[save]   Scaler      → {SCALER_PATH}")

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "scaler": scaler,
    }


# Main pipeline

def run_pipeline(raw_path: str = RAW_PATH) -> dict:
    print("=" * 60)
    print("  Online Shoppers Intention — Preprocessing Pipeline")
    print("=" * 60)

    df = load_raw(raw_path)
    validate_schema(df)
    df = handle_missing(df)
    df = remove_duplicates(df)
    df = encode_features(df)
    report_correlations(df)
    artefacts = split_and_save(df)

    print("=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)
    return artefacts

if __name__ == "__main__":
    run_pipeline()
