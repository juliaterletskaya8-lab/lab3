import pandas as pd
from src.preprocessing import fill_missing, normalize

def test_fill_missing():
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, 2, 3]})
    out = fill_missing(df, method="mean")
    assert out.isna().sum().sum() == 0

def test_normalize():
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = normalize(df.copy())
    assert round(out["a"].min(), 6) == 0.0
    assert round(out["a"].max(), 6) == 1.0
