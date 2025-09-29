import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def fill_missing(df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    """Заполняет пропуски (только для числовых столбцов)."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns
    if method == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif method == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    else:
        raise ValueError(f"Unknown method: {method}")
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Нормализует числовые столбцы в диапазон [0,1]."""
    df = df.copy()
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] == 0:
        return df
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(numeric)
    scaled_df = pd.DataFrame(scaled, columns=numeric.columns, index=df.index)
    df[numeric.columns] = scaled_df
    return df
