def accuracy_score_custom(y_true, y_pred):
    """Простая accuracy: fraction of equal labels."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)
