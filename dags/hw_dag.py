from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import sys
import os
import logging

logger = logging.getLogger(__name__)

# Добавь путь к модулям глобально
sys.path.insert(0, os.path.expanduser('~/airflow_hw'))


def train_model(**context):
    """Функция для обучения модели"""
    try:
        logger.info("Начинаем обучение модели...")

        # Импортируем модуль внутри функции
        from modules.pipeline import pipeline

        result = pipeline()
        logger.info(f"Обучение завершено. Результат: {result}")
        return result

    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {e}", exc_info=True)
        raise


def make_predictions(**context):
    """Функция для предсказаний"""
    try:
        logger.info("Начинаем предсказания...")

        # Получаем результат из предыдущей задачи
        ti = context['ti']
        train_result = ti.xcom_pull(task_ids='pipeline')
        logger.info(f"Получен результат обучения: {train_result}")

        # Импортируем модуль
        from modules.predict import predict

        result = predict()
        logger.info(f"Предсказания завершены. Результат: {result}")
        return result

    except Exception as e:
        logger.error(f"Ошибка при предсказаниях: {e}", exc_info=True)
        raise


with DAG(
        'ml_pipeline',
        start_date=datetime(2024, 1, 1),
        schedule=None,  # Только ручной запуск
        catchup=False,
        max_active_runs=1,  # Только один запуск за раз
        default_args={
            'owner': 'airflow',
            'retries': 0,  # Без ретраев для отладки
            'retry_delay': None,
        },
        tags=['ml'],
        doc_md="""ML Pipeline для обучения модели и предсказаний"""
) as dag:
    train_task = PythonOperator(
        task_id='pipeline',
        python_callable=train_model,
        # УБРАТЬ: provide_context=True,  # <-- УДАЛИ ЭТУ СТРОКУ
        execution_timeout=None,
    )

    predict_task = PythonOperator(
        task_id='predict',
        python_callable=make_predictions,
        # УБРАТЬ: provide_context=True,  # <-- УДАЛИ ЭТУ СТРОКУ
        execution_timeout=None,
    )

    train_task >> predict_task

