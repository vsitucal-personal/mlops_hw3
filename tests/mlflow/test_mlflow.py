
from mlflow_dir.model import (
    load_dataset,
    process_train_test_split,
    process_random_forest_model,
)
from loguru import logger
from sklearn.datasets import load_iris


def test_load_dataset():
    data, target, features = load_dataset(logger)
    assert data is not None
    assert target is not None
    assert features is not None


def test_process_train_test_split():
    iris = load_iris()
    X_train, X_test, y_train, y_test = process_train_test_split(iris.data, iris.target)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0


def test_process_random_forest_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = process_train_test_split(iris.data, iris.target)
    model = process_random_forest_model(logger, 100, 42, X_train, y_train)
    assert model is not None
    assert hasattr(model, "predict")
