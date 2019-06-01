# Introduction
A music streaming startup, Sparkify, has grown their user base and song database even more and want to move their data warehouse to a data lake. Their data resides in S3, in a directory of JSON logs on user activity on the app, as well as a directory with JSON metadata on the songs in their app.

As their data engineer, I was tasked with building an ETL pipeline that extracts their data from S3, processes them using Spark, and loads the data back into S3 as a set of dimensional tables. This will allow their analytics team to continue finding insights in what songs their users are listening to.

# ETL Pipeline
See file `etl.py` for the ETL code that processes the JSON logs on S3, processes them with Spark, and loads the data back into S3 as dimensional tables in Parquet format.

# Star Schema
There is one fact table named <tt>songplays</tt> and *four* dimension tables:
- <tt>users</tt>
- <tt>songs</tt>
- <tt>artists</tt>
- <tt>time</tt>

![ERD](./Data Lake ERD.png?raw=true)

# Project Datasets

## Song Dataset

The first dataset is a subset of real data from the Million Song Dataset. Each file is in JSON format and contains metadata about a song and the artist of that song. The files are partitioned by the first three letters of each song's track ID.

## Log Dataset
The second dataset consists of log files in JSON format generated by this event simulator based on the songs in the dataset above. These simulate app activity logs from an imaginary music streaming app based on configuration settings.

The log files in the dataset are partitioned by year and month.

# Running the Code
You run the ETL code as follows:
```bash
python etl.py
```
This will create a PySpark instance in standalone mode.
