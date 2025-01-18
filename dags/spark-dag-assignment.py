from datetime import timedelta
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "dibimbing",
    "retry_delay": timedelta(minutes=5),
}

spark_dag = DAG(
    dag_id="spark_airflow_dag_asg",
    default_args=default_args,
    schedule_interval=None,
    dagrun_timeout=timedelta(minutes=60),
    description="Extract Postgres and analytic",
    start_date=days_ago(1),
)

Extract = SparkSubmitOperator(
    application="/spark-scripts/spark-extract-analytic.py",
    conn_id="spark_main",
    task_id="spark_submit_task",
    dag=spark_dag,
    jars='/spark-scripts/jar/postgresql-42.7.5.jar',
    # packages="org.postgresql:postgresql:42.2.18",    # opsi bisa .jar atau package
)

Extract
