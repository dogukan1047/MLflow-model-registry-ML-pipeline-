FROM apache/airflow:latest

USER root

# Install dependencies
RUN pip install mlflow scikit-learn

USER airflow
