import pyspark
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql import Window

postgres_host = "dataeng-postgres"
postgres_dw_db = "warehouse"
postgres_user = "user"
postgres_password = "password"

sparkcontext = pyspark.SparkContext.getOrCreate(conf=(pyspark.SparkConf().setAppName("Dibimbing")))
sparkcontext.setLogLevel("WARN")
spark = pyspark.sql.SparkSession(sparkcontext.getOrCreate())

jdbc_url = f"jdbc:postgresql://{postgres_host}/{postgres_dw_db}"
jdbc_properties = {
    "user": postgres_user,
    "password": postgres_password,
    "driver": "org.postgresql.Driver",
    "stringtype": "unspecified",
}
print(jdbc_properties)

df = spark.read.jdbc(
    jdbc_url, 
    "public.retail", 
    properties=jdbc_properties
)

df_sales_by_month = (
    df.groupBy(F.date_format('invoiceDate', 'MMMM').alias('month'))
    .agg(
        F.sum('quantity').alias('sales'),
        F.sum(df.quantity * df.unitprice).alias('revenue')
    )
    .orderBy(F.desc('revenue'))
    .withColumn('revenue', F.format_string("$%,.2f", F.col('revenue')))
)

df_sales_by_day = (
    df.groupBy(F.date_format('invoicedate', 'EEEE').alias('day'))
    .agg(
        F.sum('quantity').alias('sales'),
        F.sum(df.quantity * df.unitprice).alias('revenue')
    )
    .orderBy(F.desc('revenue'))
    .withColumn('revenue', F.format_string("$%,.2f", F.col('revenue')))
)

df_product_sales_revenue = (
    df.groupBy('description')
    .agg(
        F.sum('quantity').alias('sales'),
        F.sum(df.quantity * df.unitprice).alias('revenue')
    )
    .withColumnRenamed('description','product')
    .orderBy(F.desc("revenue"))   
    .withColumn('revenue', F.format_string("$%,.2f", F.col('revenue')))  
)

window_rank = Window.orderBy(F.desc('revenue')) 
df_country_sales_revenue = (
    df.groupBy('country')
    .agg(
        F.sum('quantity').alias('sales'),
        F.sum(df.quantity * df.unitprice).alias('revenue')
    )
    .withColumn('rank', F.rank().over(window_rank))
    .orderBy(F.desc("revenue"))   
    .withColumn('revenue', F.format_string("$%,.2f", F.col('revenue')))  
)

window_partition = Window.partitionBy('customerid').orderBy('invoicedate')
df_customer_retention = (
    df.withColumn('purchase_date', F.to_date(df.invoicedate))  
    .withColumn('purchase_month', F.date_trunc('MONTH', 'purchase_date'))
    .withColumn('first_purchase_month', F.date_trunc('MONTH', F.min('purchase_date').over(window_partition)))  
    .withColumn('cohort_month', F.months_between('purchase_month', 'first_purchase_month').cast('int'))  
    .withColumn('cohort_month_label', F.concat(F.lit('month_'), F.col('cohort_month').cast('string')))  
    .groupBy('first_purchase_month', 'cohort_month_label')  
    .agg(F.countDistinct('customerid').alias('users'))  
    .groupBy(F.date_format('first_purchase_month', 'yyyy-MM').alias('first_purchase'))  
    .pivot('cohort_month_label')  
    .agg(F.first('users'))  
)

cohort_columns = ['first_purchase'] + [f'month_{i}' for i in range(13)]  
df_pivot_customer_retention = df_customer_retention.select(cohort_columns).orderBy('first_purchase')

df_sales_by_month.show()
df_sales_by_day.show()
df_product_sales_revenue.show(20)
df_country_sales_revenue.show(20)

(
    df_pivot_customer_retention
    .write
    .mode("overwrite")
    .option("truncate", "true")
    .jdbc(
        jdbc_url,
        'public.customer_retention_sample',
        properties=jdbc_properties
    )
)

(
    spark
    .read
    .jdbc(
        jdbc_url,
        'public.customer_retention_sample',
        properties=jdbc_properties
    )
    .show()
)
