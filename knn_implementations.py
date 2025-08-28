import heapq, math
from collections import Counter
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window


def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def map_function(train_partition, test_data, k):
    """
    Map function following Algorithm 1
    """
    train_list = list(train_partition)
    results = []
    
    # For each test instance
    for test_idx, (test_features, _) in enumerate(test_data):
        # Compute k-NN for this test point against this training partition
        distances = []
        for train_features, train_label in train_list:
            dist = euclidean_distance(test_features, train_features)
            distances.append((dist, train_label))
        
        # Get k nearest neighbors from this partition
        k_nearest = heapq.nsmallest(k, distances, key=lambda x: x[0])
        
        # Create candidate set CDj for this partition
        cd_j = [(train_label, dist) for dist, train_label in k_nearest]
        
        # Emit with test_idx as key and partition results
        results.append((test_idx, cd_j))
    
    return results

def reduce_operation(cd_list, k):
    """
    Reduce operation following Algorithm 2
    Merge candidate sets and keep k best neighbors
    """
    all_candidates = []
    for cd_j in cd_list:
        all_candidates.extend(cd_j)
    
    # Sort by distance and take k nearest
    # heapq.nsmallest is like the nested loops in the paper, but uses python's built-in heap data structure
    k_nearest = heapq.nsmallest(k, all_candidates, key=lambda x: x[1])
    return k_nearest

def majority_voting(neighbors):
    """
    Cleanup process following Algorithm 3
    """
    classes = [label for label, _ in neighbors]
    return Counter(classes).most_common(1)[0][0]

def knn_rdd(sc, train_data, test_data, k, num_partitions=2):
    """
    RDD-based kNN following the paper's algorithm more closely
    """

    # Create training RDD with partition indices
    training_rdd = sc.parallelize(train_data, num_partitions)
    
    # Map phase: process each test point against each training partition
    mapped_rdd = training_rdd.mapPartitions(
        lambda partition: map_function(partition, test_data, k)
    )
    
    # Reduce phase: combine results for each test point
    reduced_rdd = mapped_rdd.groupByKey().mapValues(
        lambda cd_lists: reduce_operation(list(cd_lists), k)
    )
    
    # Cleanup phase: majority voting
    predictions = reduced_rdd.mapValues(majority_voting)
    return predictions.collect()

def knn_df(spark, train_data, test_data, k, num_partitions=2):
    """
    DataFrame implementation following the same 3-phase structure as RDD version
    """
    # Create DataFrames with distinct column names
    train_df = spark.createDataFrame(train_data, ["id_train", "train_features", "label"])
    test_df = spark.createDataFrame(test_data, ["id_test", "test_features"])
    
    # Repartition training data
    train_df = train_df.repartition(num_partitions).withColumn("partition_id", F.spark_partition_id())
    
    # PHASE 1: MAP - Get k-NN per partition (like map_function)
    cross_joined = test_df.crossJoin(train_df)
    
    # Calculate distances
    with_distances = cross_joined.withColumn(
        "distance",
        F.sqrt(F.aggregate(
            F.arrays_zip("test_features", "train_features"),
            F.lit(0.0),
            lambda acc, x: acc + F.pow(x.getField("test_features") - x.getField("train_features"), 2)
        ))
    )
    
    # Get k nearest per partition (MAP output - CDj sets)
    partition_window = Window.partitionBy("id_test", "partition_id").orderBy("distance")
    map_output = with_distances.withColumn(
        "rank_in_partition", 
        F.row_number().over(partition_window)
    ).filter(F.col("rank_in_partition") <= k).select("id_test", "label", "distance")
    
    # PHASE 2: REDUCE - Merge all CDj sets and keep k best (like reduce_operation)
    global_window = Window.partitionBy("id_test").orderBy("distance")
    reduce_output = map_output.withColumn(
        "global_rank",
        F.row_number().over(global_window)
    ).filter(F.col("global_rank") <= k)
    
    # PHASE 3: CLEANUP - Majority voting (like majority_voting)
    vote_window = Window.partitionBy("id_test").orderBy(F.desc("vote_count"))
    predictions = (
        reduce_output
        .groupBy("id_test", "label")
        .count()
        .withColumnRenamed("count", "vote_count")
        .withColumn("rank", F.row_number().over(vote_window))
        .filter(F.col("rank") == 1)
        .select("id_test", F.col("label").alias("predicted_class"))
    )
    
    return predictions.collect()


if __name__ == "__main__":
    spark = SparkSession.builder.appName("KNN_Paper_Implementation").getOrCreate()
    sc = spark.sparkContext

    # Test data
    train = [(i, [1.0, 2.0], "A") for i in range(2)] + [(i+2, [5.0, 8.0], "B") for i in range(2)]
    test = [(0, [1.2, 1.9]), (1, [5.5, 8.5])]
    
    train_rdd = [([1.0, 2.0], "A"), ([1.5, 1.8], "A"), ([5.0, 8.0], "B"), ([6.0, 9.0], "B")]
    test_rdd = [([1.2, 1.9], None), ([5.5, 8.5], None)]

    print("Paper-based RDD implementation:")
    result_rdd = knn_rdd(sc, train_rdd, test_rdd, k=2)
    print(result_rdd)

    print("\nPaper-based DataFrame implementation:")
    result_df = knn_df(spark, train, test, k=2, num_partitions=2)
    result_df.show()