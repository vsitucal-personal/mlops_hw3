# Homework 3
 
## Docker compose setup
```commandline
docker-compose --env-file sample_env up --build -d
```
Docker Desktop UI:
![ui](images/ui.png)


## Mlflow link
- Access with [http://localhost:5000](http://localhost:5000)
  ![1](images/1.png)

## Minio Object Storage
- Access with [http://localhost:9001](http://localhost:9001)
  ![mini](images/mini.png)

## Jupyterlab link
- Access with [http://localhost:8888?token=easy](http://localhost:8888?token=easy)
- Run `mlflow/notebooks/train_and_register_model.ipynb` to train and register model in mlflow
- Model should be in mlflow after 
  ![2](images/2.png)
  ![3](images/3.png)

## Test Fastapi endpoint
- [http://localhost:8000/docs](http://localhost:8000/docs) fro OpenAPI docs
- [http://localhost:8000/predict](http://localhost:8000/predict) for `/predict` endpoint
- Run `mlflow/notebooks/predict.ipynb` to predict
  ![4](images/4.png)

## Pytest code coverage
- [http://localhost:7000](http://localhost:7000)

## Cleanup
```commandline
docker compose down
```
  