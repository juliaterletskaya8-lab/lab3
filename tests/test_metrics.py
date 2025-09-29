from src.metrics import accuracy_score_custom

def test_accuracy():
    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 1, 0]
    acc = accuracy_score_custom(y_true, y_pred)
    assert acc == 0.75
