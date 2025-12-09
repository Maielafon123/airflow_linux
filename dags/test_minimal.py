from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def simple_task():
    logger.info("Минимальная задача выполняется!")
    return "УСПЕХ"

with DAG(
    dag_id="test_minimal",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        'owner': 'airflow',
        'retries': 0,
    },
) as dag:
    
    task = PythonOperator(
        task_id="simple_task",
        python_callable=simple_task,
    )
