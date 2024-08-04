from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

default_args = {
    'owner': 'Dodo',
    'depends_on_past': False,
    'start_date': datetime(2024, 5, 7),
    'schedule_interval':'@daily',
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'mlflow_airflow_complex_example',
    default_args=default_args,
    description='Basic ML pipeline with MLflow and Airflow',
    schedule_interval=timedelta(days=1),
)

def preprocess_data():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def train_model(**kwargs):
    ti = kwargs['ti']
    X_train, X_test, y_train, y_test = ti.xcom_pull(task_ids='preprocess_data')

    param_grid = {'logisticregression__C': [0.1, 1.0, 10.0]}
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logisticregression', LogisticRegression(max_iter=200))
    ])
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    accuracy = accuracy_score(y_test, best_model.predict(X_test))
    precision = precision_score(y_test, best_model.predict(X_test), average='weighted')
    recall = recall_score(y_test, best_model.predict(X_test), average='weighted')
    f1 = f1_score(y_test, best_model.predict(X_test), average='weighted')
    
    ti.xcom_push(key='best_model', value=best_model)
    ti.xcom_push(key='accuracy', value=accuracy)
    ti.xcom_push(key='precision', value=precision)
    ti.xcom_push(key='recall', value=recall)
    ti.xcom_push(key='f1', value=f1)

def log_metrics(**kwargs):
    ti = kwargs['ti']
    best_model = ti.xcom_pull(task_ids='train_model', key='best_model')
    accuracy = ti.xcom_pull(task_ids='train_model', key='accuracy')
    precision = ti.xcom_pull(task_ids='train_model', key='precision')
    recall = ti.xcom_pull(task_ids='train_model', key='recall')
    f1 = ti.xcom_pull(task_ids='train_model', key='f1')
    
    with mlflow.start_run():
        mlflow.log_param("model_type", "Logistic Regression")
        mlflow.log_param("max_iter", 200)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(best_model, "model")

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag,
)

log_metrics_task = PythonOperator(
    task_id='log_metrics',
    python_callable=log_metrics,
    provide_context=True,
    dag=dag,
)

preprocess_data_task >> train_model_task >> log_metrics_task
