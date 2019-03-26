---
title: Optimizing SQL Statements
date: 2017-03-27 10:32:50
tags: 
- MySQL
- Data Warehousing
category: 
- 时习之
- Miscellaneous
description: MySQL Query Optimization Notes for Database of One Billion Entries
---

## PART I Database Level

### Choice of Storage Engine

The most commonly used storage engines are:

* **InnoDB**: Default setting for MySQL. 

  **Pros:**

  * **A transactional storage engine**.  That means it supports rollback / commit (All or Nothing), rather writing every change directly into disk (though by default auto commit is on, but you can use `SET AUTOCOMMIT=0;` ). 

  * **Row-level locking**. That means, it allows concurrent access to one table. Better to use when you will have multiple read/ write sections simultaneously.

  * **Foreign key constraints**. It automatically checks foreign integrity.

  * **Supports large buffer pool for both DATA and index**. While MyISAM only supports buffer for index.

  **Cons:**

  * Can't be compressed for fast access in system table space. But it could have compress tables in general table space or file per table space.

  * No full text indexing

* **MyISAM**: a non-transactional storage engine. 

  **Pros:**

  * **It is designed for FAST READ**. In a situation that the read is frequent while the write is little (read-write-ratio<15%). Especially good for extensive SELECT queries. It's frequently used in data warehousing. 

  * **Full text indexing**

  **Cons:**

  * **No foreign key constraints.**

  * **Non-transactional**. thus no roll-back capability.

  * **Row limit**: $2^{32}$, about 4.3 billion records at maximum.

Besides, other available storage engines include:

* **CSV engine**: stores data in CSV. Easily integrated with other applications.

* **Archive storage engine**: optimized for high speed inserting task.



To see support engines in the database, 

```mysql
show storage engines;
```


To see engines currently used for each table in the database,

```mysql
SHOW TABLE STATUS FROM $YourDatabaseName\G
```

Sample Output:

```
mysql> SHOW TABLE STATUS FROM airline\G
*************************** 1. row ***************************
           Name: DIM_AIRLINE
         Engine: InnoDB
        Version: 10
     Row_format: Dynamic
           Rows: 16256
 Avg_row_length: 97
    Data_length: 1589248
Max_data_length: 0
   Index_length: 1835008
      Data_free: 0
 Auto_increment: NULL
    Create_time: 2017-03-18 21:54:52
    Update_time: NULL
     Check_time: NULL
      Collation: latin1_swedish_ci
       Checksum: NULL
 Create_options: 
        Comment: 
*************************** 2. row ***************************
           Name: DIM_AIRPORT
         Engine: InnoDB
(and more)
```

To alter engine,

```mysql
ALTER TABLE $YourTableName ENGINE='MyISAM';
```



### Optimizing Buffering and Caching

The caching size should be large enough to hold frequently queried data and smaller than the memory size of the instance. 

* **InnoDB**: Default caching size: 128 MB

  Show the current caching size:

  ```mysql
  SELECT @@innodb_buffer_pool_size/1024/1024/1024;
  ```

  Sample results:

  ```
  +------------------------------------------+
  | @@innodb_buffer_pool_size/1024/1024/1024 |
  +------------------------------------------+
  |                           0.125000000000 |
  +------------------------------------------+
  1 row in set (0.00 sec)
  ```

  Change the caching size:

  ```mysql
  mysqld --innodb_buffer_pool_size=8G --innodb_buffer_pool_instances=16
  ```





* **MyISAM**: Key cache buffer is designed for index only and allows concurrent access. 

  Show the current key cache buffer size:

  ```
  mysql> SHOW VARIABLES LIKE 'key_buffer_size';
  +-----------------+----------+
  | Variable_name   | Value    |
  +-----------------+----------+
  | key_buffer_size | 16777216 |
  +-----------------+----------+
  1 row in set (0.00 sec)
  ```

  Change the caching size:

  ```
  SET GLOBAL key_buffer_size = 26777218;
  ```

  ​

  A suggested strategy for MyISAM is, though, to divide cache into three parts: hot cache for read intensive index, cold cache for write intensive tables, and a default warm cache for other operations.

  ​

* **MySQL Query Cache**: Stores query results. Useful when data tables doesn't change a lot and queries are similar, such as web services.

  Show the current query cache size:

  ```
  mysql> SHOW VARIABLES LIKE 'query_cache_size';
  +------------------+----------+
  | Variable_name    | Value    |
  +------------------+----------+
  | query_cache_size | 16777216 |
  +------------------+----------+
  1 row in set (0.00 sec)
  ```

  Change the caching size:

  ```
  SET GLOBAL query_cache_size = 46777216;
  ```




## PART II Table Creation

### Table Compression

Table compression is very useful for read-intensive applications. It helps particularly when there is a lot of character string columns. Why?

> "Because data compression enables smaller database size, reduced I/O, and improved throughput, at the small cost of increased CPU utilization."



Two ways to use compressed table for InnoDB:

* Upon table creation

  As compression is not enabled in the system table space (that is where there contains system files and we normally create tables), we need to create a general table space first.

  ```
  -- Create a table space
  CREATE TABLESPACE `ts2` ADD DATAFILE 'ts2.ibd' FILE_BLOCK_SIZE = 8192 Engine=InnoDB;
  -- Create a compressed table
  CREATE TABLE t4 (c1 INT PRIMARY KEY) TABLESPACE ts2 ROW_FORMAT=COMPRESSED KEY_BLOCK_SIZE=8;
  ```

* Alter table

  ```
  Alter table t4 ROW_FORMAT=COMPRESSED;
  ```



For MyISAM engine, compressed table is generated with [myisampack](https://dev.mysql.com/doc/refman/5.5/en/myisampack.html) tool.



### Right Indexing

With clustered index, the table is sorted when being stored on disk. A non-clustered index is a row locator (RiD), it's good to use in small table, but it finds a row required iterating through the whole table which is very bad for large table. 

In default, MySQL makes primary key as clustered index. Else MySQL will find the first all not null column as index (might be pretty long though!). 

According to MySQL documentation, 

> "Accessing a row through the clustered index is fast because the index search leads directly to the page with all the row data. If a table is large, the clustered index architecture often saves a disk I/O operation when compared to storage organizations that store row data using a different page from the index record. (For example, `MyISAM` uses one file for data rows and another for index records.)"



Other than clustered indexing (primary key), we could still set secondary indexing. The indexing will then contain **data** for the column which it indexes to, and the primary key.



As you can see, without indexing, where clause has to go over the whole table to look up value.

```
mysql> explain select avg(carrier_delay) from FACT_FLIGHT_3_SAMPLE where unique_carrier = 'b6' group by unique_carrier\G
*************************** 1. row ***************************
           id: 1
  select_type: SIMPLE
        table: FACT_FLIGHT_3_SAMPLE
   partitions: NULL
         type: ALL
possible_keys: NULL
          key: NULL
      key_len: NULL
          ref: NULL
         rows: 50000
     filtered: 10.00
        Extra: Using where
1 row in set, 1 warning (0.00 sec)

```



When indexing is created for the column of interests, the where clause is executed without accessing the actual table.

```
mysql> CREATE INDEX carrier_ix ON FACT_FLIGHT_3_SAMPLE (UNIQUE_CARRIER ASC);
Query OK, 50000 rows affected (0.10 sec)
Records: 50000  Duplicates: 0  Warnings: 0

mysql> explain select avg(carrier_delay) from FACT_FLIGHT_3_SAMPLE where unique_carrier = 'b6' group by unique_carrier\G
*************************** 1. row ***************************
           id: 1
  select_type: SIMPLE
        table: FACT_FLIGHT_3_SAMPLE
   partitions: NULL
         type: ref
possible_keys: carrier_ix
          key: carrier_ix
      key_len: 9
          ref: const
         rows: 1
     filtered: 100.00
        Extra: NULL
1 row in set, 1 warning (0.00 sec)
```

According to experiment on data table with one billion entries, the time spent with indexing is only 50% without indexing.



## PART III Query Level

### Use EXPLAIN

Explain is used to show a query execution plan. It helps understand what we are trying to do: how many operations are needed and how many rows are affected. Examples could be found in previous section.



### Queries

* Isolate
>  "Isolate and tune any part of the query, so that a function could be called once for every row in the result set rather than in the whole table."

* Aggregate functions:
>  "The most efficient way to process `GROUP BY` is when an index is used to directly retrieve the grouping columns. With this access method, MySQL uses the property of some index types that the keys are ordered (for example, `BTREE`). This property enables use of lookup groups in an index without having to consider all keys in the index that satisfy all `WHERE` conditions. " 




## Reference

* [MySQL Storage Engines](http://zetcode.com/databases/mysqltutorial/storageengines/)
* [When to use MyISAM and InnoDB](http://stackoverflow.com/questions/15678406/when-to-use-myisam-and-innodb)
* [SQL query performance killers – understanding poor database indexing](https://www.sqlshack.com/sql-query-performance-killers-understanding-poor-database-indexing/)
* [Optimization](https://dev.mysql.com/doc/refman/5.7/en/optimization.html)

Note* All quotes are from Reference 4 Optimization - the official documentation of MySQL's.