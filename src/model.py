import numpy as np
from sklearn.linear_model import LogisticRegression

def train_model(X, y):
    """Обучает простую логистическую регрессию и возвращает модель."""
    X = np.array(X)
    y = np.array(y)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model
