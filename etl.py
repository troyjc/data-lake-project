import configparser
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import TimestampType, DateType
import boto3
import pathlib

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def json_files(bucket, prefix):
    """Use the boto3 client API to return a list of JSON files.

    Args:
        bucket (string): Name of the bucket to list.
        prefix (string): Limits the response to keys that begin with the
            specified prefix.
        
    Yields:
        string: The s3a path to the JSON file.
    """
    s3 = boto3.client('s3')
    kwargs = {'Bucket': bucket, 'Prefix': prefix}
    while True:
        response = s3.list_objects_v2(**kwargs)
        for object in response['Contents']:
            key = object['Key']
            if pathlib.Path(key).suffix != '.json':
                continue  # Filter to only return JSON files
            yield 's3a://{}/{}'.format(bucket, key)

        # See if there are more results to fetch
        if response['IsTruncated']:
            kwargs['ContinuationToken'] = response['NextContinuationToken']
        else:
            return
        

def bucket_name(path):
    """Returns the bucket name from the S3 URL."""
    return pathlib.Path(path).parts[1]


def file_contents(bucket, key):
    """Return the content (body) associated with the specified S3 key."""
    s3 = boto3.client('s3')
    return s3.get_object(Bucket=bucket, Key=key)['Body'].read().decode('UTF-8')


def process_song_data(spark, input_data, output_data):
    """Process the JSON files in the song dataset.
    
    Creates the 'songs' and 'artists' tables as Parquet files.
    
    Args:
        spark (SparkSession): Main entry point for DataFrame and SQL functionality.
        input_data (string): S3 URL with the bucket name
        output_data (string): S3 URL with the bucket and prefix names
    """

    # Get filepath to song data files
    # It seems like a good idea to use "song_data/*/*/*/*.json"), but it's not because
    # S3 isn't a file system. Instead, use boto3 to get the list of JSON files
    bucket = bucket_name(input_data)
    files = list(json_files(bucket, prefix='song_data'))
    
    # Read the song data files
    df = spark.read.json(files)    
    
    df.registerTempTable("song_data")

    # Extract columns to create songs table
    songs_table = spark.sql("SELECT DISTINCT song_id, title, artist_id, year, duration FROM song_data")

    # Write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy("year", "artist_id").parquet(os.path.join(output_data, 'songs'))

    # Extract columns to create artists table
    artists_table = spark.sql("""SELECT DISTINCT artist_id,
                                                 artist_name AS name,
                                                 artist_location AS location,
                                                 artist_latitude AS latitude,
                                                 artist_longitude AS longitude
                                 FROM song_data""")

    # Write artists table to parquet files
    artists_table.write.parquet(os.path.join(output_data, 'artists'))


def process_log_data(spark, input_data, output_data):
    """Process the JSON files in the log dataset.
    
    Creates the 'users', 'time', and 'songplays' tables as Parquet files.
    
    Args:
        spark (SparkSession): Main entry point for DataFrame and SQL functionality.
        input_data (string): S3 URL with the bucket name
        output_data (string): S3 URL with the bucket and prefix names
    """

    # Get filepath to log data files
    bucket = bucket_name(input_data)
    files = list(json_files(bucket, prefix='log_data'))
    print("Number of log files is {}".format(len(files)))
    
    # Read the log data files
    df = spark.read.json(files)
    
    # Filter by actions for song plays
    df = df.filter("page = 'NextSong'")

    df.registerTempTable("log_data")

    # Extract columns for users table    
    users_table = spark.sql("""SELECT DISTINCT userId AS user_id,
                                               firstName AS first_name,
                                               lastName AS last_name,
                                               gender,
                                               level
                               FROM log_data""")
    
    # Write users table to parquet files
    users_table.write.parquet(os.path.join(output_data, 'users'))

    # Create datetime column from original timestamp column
    df = df.withColumn("datetime", (col("ts") / 1000).cast(TimestampType()))
    
    # Create timestamp column from original timestamp column
    df = df.withColumn("timestamp", to_timestamp("datetime"))
    
    df.registerTempTable("log_data")

    # Extract columns to create time table
    time_table = spark.sql("""SELECT datetime AS start_time,
                                     hour(datetime) AS hour,
                                     day(datetime) AS day,
                                     weekofyear(datetime) AS week,
                                     month(datetime) AS month,
                                     year(datetime) AS year,
                                     dayofweek(datetime) AS weekday
                              FROM log_data""")
    
    # Write time table to parquet files partitioned by year and month
    time_table.write.partitionBy("year", "month").parquet(os.path.join(output_data, 'time'))

    # Read in song data to use for songplays table
    song_df = spark.read.parquet(os.path.join(output_data, 'songs'))
    song_df.registerTempTable("songs")

    # Extract columns from joined song and log datasets to create songplays table 
    songplays_table = spark.sql("""SELECT monotonically_increasing_id() AS songplay_id,
                                          datetime AS start_time,
                                          userId AS user_id,
                                          level,
                                          song_id,
                                          artist_id,
                                          sessionId AS session_id,
                                          location,
                                          userAgent AS user_agent,
                                          year(datetime) AS year,
                                          month(datetime) AS month
                                   FROM log_data JOIN songs
                                   WHERE log_data.song = songs.title""")

    # Write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy("year", "month").parquet(os.path.join(output_data, 'songplays'))


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://troych-udacity/analytics/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()

