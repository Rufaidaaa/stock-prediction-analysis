#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import time

def main():
    # Initialize Spark session with Hive support
    spark = SparkSession.builder \
        .appName("StockMarketAnalysis") \
        .enableHiveSupport() \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    try:
        # =============================================
        # 1. Load Data from HDFS
        # =============================================
        print("\n[1/4] Loading data from HDFS...")
        start_time = time.time()
        
        # Read from HDFS path - use the correct path format for Dataproc
        df = spark.read.csv("/user/stockdata/merged_stock_data.csv", header=True, inferSchema=True)
        
        # Convert data types and handle missing values
        df = df.withColumn("Date", to_date(col("Date"), "M/d/yyyy")) \
               .withColumn("Low", col("Low").cast("double")) \
               .withColumn("Open", col("Open").cast("double")) \
               .withColumn("Volume", col("Volume").cast("long")) \
               .withColumn("High", col("High").cast("double")) \
               .withColumn("Close", col("Close").cast("double")) \
               .withColumn("Adjusted Close", col("Adjusted Close").cast("double")) \
               .na.fill(0, subset=["Volume"]) \
               .na.fill(df.select(mean("Close")).first()[0], subset=["Close"])
        
        load_time = time.time() - start_time
        print(f"Data loaded in {load_time:.2f} seconds")
        print(f"Total records: {df.count():,}")
        
        # =============================================
        # 2. Data Processing
        # =============================================
        print("\n[2/4] Performing data processing...")
        process_start = time.time()
        
        # Calculate derived metrics
        df_processed = df.withColumn("PriceChange", col("Close") - col("Open")) \
                        .withColumn("PercentChange", round((col("Close") - col("Open")) / col("Open") * 100, 2)) \
                        .withColumn("Year", year(col("Date"))) \
                        .withColumn("Month", month(col("Date"))) \
                        .withColumn("Volatility", round(col("High") - col("Low"), 4))
        
        process_time = time.time() - process_start
        print(f"Processing completed in {process_time:.2f} seconds")
        
        # =============================================
        # 3. Analytical Queries
        # =============================================
        print("\n[3/4] Running analytical queries...")
        analysis_start = time.time()
        
        # Query 1: Yearly statistics
        yearly_stats = df_processed.groupBy("Year") \
            .agg(
                round(avg("Close"), 2).alias("AvgClose"),
                round(avg("Volatility"), 4).alias("AvgVolatility"),
                sum("Volume").alias("TotalVolume"),
                round(max("High"), 2).alias("YearHigh"),
                round(min("Low"), 2).alias("YearLow")
            ).orderBy("Year")
        
        # Query 2: Moving Average (30-day window)
        windowSpec = Window.orderBy("Date").rowsBetween(-29, 0)
        df_with_ma = df_processed.withColumn("30DayMA", round(avg("Close").over(windowSpec), 2))
        
        # Query 3: Most volatile days
        most_volatile = df_processed.orderBy(col("Volatility").desc()).limit(5)
        
        analysis_time = time.time() - analysis_start
        print(f"Analysis completed in {analysis_time:.2f} seconds")
        
        # =============================================
        # 4. Save Results
        # =============================================
        print("\n[4/4] Saving results...")
        save_start = time.time()
        
        # Save to GCS with header and single file (using coalesce)
        yearly_stats.coalesce(1) \
            .write.mode("overwrite") \
            .option("header", "true") \
            .csv("gs://bigdatabucket123/output/yearly_stats")

        df_with_ma.coalesce(1) \
            .write.mode("overwrite") \
            .option("header", "true") \
            .csv("gs://bigdatabucket123/output/with_moving_avg")

        most_volatile.coalesce(1) \
            .write.mode("overwrite") \
            .option("header", "true") \
            .csv("gs://bigdatabucket123/output/most_volatile_days")
        
        save_time = time.time() - save_start
        print(f"Results saved in {save_time:.2f} seconds")
        
        # =============================================
        # 5. Performance Metrics
        # =============================================
        total_time = time.time() - start_time
        print("\nPerformance Summary:")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Breakdown:")
        print(f"- Data loading: {load_time:.2f} seconds ({load_time/total_time*100:.1f}%)")
        print(f"- Processing: {process_time:.2f} seconds ({process_time/total_time*100:.1f}%)")
        print(f"- Analysis: {analysis_time:.2f} seconds ({analysis_time/total_time*100:.1f}%)")
        print(f"- Saving results: {save_time:.2f} seconds ({save_time/total_time*100:.1f}%)")
        
        print("\nResults saved to:")
        print("- HDFS: /user/stockdata/output/")
        print("- GCS: gs://bigdatabucket123/output/ and gs://bigdatabucket123/csv_output/")
        
    except Exception as e:
        print(f"\nError encountered: {str(e)}")
        raise e
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()