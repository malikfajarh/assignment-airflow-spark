import pyspark

postgres_host = "dataeng-postgres"
postgres_dw_db = "warehouse"
postgres_user = "user"
postgres_password = "password"

spark_host = "spark://dibimbing-dataeng-spark-master:7077"

sparkcontext = pyspark.SparkContext.getOrCreate(conf=(
        pyspark
        .SparkConf()
        .setAppName('Dibimbing')
        .setMaster(spark_host)
        .set("spark.jars", "/opt/bitnami/spark/jars/postgresql-42.2.18.jar")
    ))
sparkcontext.setLogLevel("WARN")

spark = pyspark.sql.SparkSession(sparkcontext.getOrCreate())

retail_df = spark.read.jdbc(
    jdbc_url, 
    "public.retail", 
    properties=jdbc_properties
)

retail_df.show(5)
