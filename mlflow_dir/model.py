import os
import shutil
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from loguru import logger


def load_dataset(logger):
    try:
        iris = load_iris()
        return iris.data, iris.target, iris.feature_names
    except Exception as e:
        logger.info(e)
    return None


def process_train_test_split(data, target):
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, random_state=42
    )
    return X_train, X_test, y_train, y_test


def process_random_forest_model(logger, n_estimators, random_state, x_train, y_train):
    # Train model
    model = None
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(x_train, y_train)
    except Exception as e:
        logger.info(e)
    return model


def experiment_run(experiment_name):
    logger.add("mlflow_training.log", rotation="1 MB", level="INFO")

    # Set MLflow tracking URI
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", )  # Default if not set # talks to mlflow docker
    mlflow.set_tracking_uri(mlflow_uri)

    mlflow.set_experiment(experiment_name)
    logger.info(f"Experiment set: {experiment_name}")

    # Load dataset
    ds = load_dataset(logger)
    x_train, x_test, y_train, y_test = process_train_test_split(ds[0], ds[1])

    # Define hyperparameters
    n_estimators = 100
    random_state = 42

    # Train model
    model = process_random_forest_model(logger, n_estimators, random_state, x_train, y_train)

    # Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Run started: {run_id}")

        # Log hyperparameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        logger.info(f"Logged hyperparameters: n_estimators={n_estimators}, random_state={random_state}")

        # Log metrics
        train_accuracy = model.score(x_train, y_train)
        test_accuracy = model.score(x_test, y_test)

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        logger.info(f"Logged metrics: train_accuracy={train_accuracy}, test_accuracy={test_accuracy}")

        # Log model
        model_uri = "models:/iris_classifier"
        mlflow.sklearn.log_model(model, "model")
        logger.info(f"Model logged to MLflow")

        # Register model
        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name="iris_classifier"
        )
        logger.info(f"Model registered: {result.name}, version: {result.version}")

        # Save artifacts
        artifact_path = "artifacts"
        os.makedirs(artifact_path, exist_ok=True)
        artifact_file = os.path.join(artifact_path, "iris_features.txt")
        with open(artifact_file, "w") as f:
            f.write(str(ds[2]))

        mlflow.log_artifact(artifact_file)
        logger.info(f"Artifact logged: {artifact_file}")

    logger.info("MLflow training run completed.")