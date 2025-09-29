from src.model import train_model

def test_model_training():
    X = [[0, 1], [1, 0], [1, 1], [0, 0]]
    y = [0, 1, 1, 0]
    model = train_model(X, y)
    assert hasattr(model, "predict")
    pred = model.predict([[1, 0]])[0]
    assert pred in [0, 1]
