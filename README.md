# Homework 3
 
## Docker compose setup
```commandline
docker-compose up --build -d
```
Docker Desktop UI:
![ui](images/ui.png)


## Mlflow link
- Access with [http://localhost:5000](http://localhost:5000)
  ![1](images/1.png)

## Miniio Object Storage
- Access with [http://localhost:9001](http://localhost:9001)
  ![mini](images/mini.png)

## Jupyterlab link
- Access with [http://localhost:8888?token=easy](http://localhost:8888?token=easy)
- Run `mlflow/notebooks/train_and_register_model.ipynb` to train and register model in mlflow
- Model should be in mlflow after 
  ![2](images/2.png)
  ![3](images/3.png)

## Test Fastapi endpoint
- Run `mlflow/notebooks/predict.ipynb` to predict
  ![4](images/4.png)

## Cleanup
```commandline
docker compose down
```
  