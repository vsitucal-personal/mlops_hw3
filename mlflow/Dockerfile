FROM mlflow:latest

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN useradd -m -s /bin/bash mlflowuser && \
    chown -R mlflowuser:mlflowuser /app
USER mlflowuser

