from typing import List, Dict
import pandas as pd

def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error loading CSV file at {file_path}: {e}")

def load_multiple_csv(file_paths: List[str]) -> Dict[str, pd.DataFrame]:
    """Load multiple CSV files into a dictionary of DataFrames."""
    dataframes = {}
    for path in file_paths:
        dataframes[path] = load_csv(path)
    return dataframes

def load_file(file_path: str):
    """Load a file based on its extension."""
    if file_path.endswith('.csv'):
        return load_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type for {file_path}")
