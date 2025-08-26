import math
from collections import defaultdict
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import *


def map_parameters(record):
    features, class_label = record
    results = []
    for attr_idx, attr_value in enumerate(features):
        key = f"attr-{attr_idx}-{attr_value}-{class_label}"
        results.append((key, 1))
    return results

def calculate_priors(class_counts, total_count):
    return {cls: count / total_count for cls, count in class_counts.items()}

def calculate_likelihoods(parameter_counts, class_counts):
    likelihoods = defaultdict(dict)
    for key, count in parameter_counts.items():
        parts = key.split('-')
        attr_idx, attr_value, class_label = parts[1], parts[2], parts[3]
        
        # Add Laplace smoothing (+1)
        likelihood = (count + 1) / (class_counts[class_label] + len(set(parts[2] for parts in [k.split('-') for k in parameter_counts.keys()])))
        
        attr_key = f"attr-{attr_idx}-{attr_value}"
        likelihoods[class_label][attr_key] = likelihood
        
    return likelihoods

def classify_instance(features, priors, likelihoods, class_counts):
    class_scores = {}
    
    for class_label in priors:
        # Start with log prior
        log_score = math.log(priors[class_label])
        
        # Add log likelihoods
        for attr_idx, attr_value in enumerate(features):
            attr_key = f"attr-{attr_idx}-{attr_value}"
            if attr_key in likelihoods[class_label]:
                log_score += math.log(likelihoods[class_label][attr_key])
            else:
                # Laplace smoothing for unseen attribute values
                log_score += math.log(1 / (class_counts[class_label] + 2))
        
        class_scores[class_label] = log_score
    
    return max(class_scores.items(), key=lambda x: x[1])[0]

# Classify test data

def naive_bayes_rdd(sc, train_data, test_data):
    """
    RDD-based Naive Bayes following the paper's MapReduce approach
    
    train_data: list of (features_list, class_label)
    test_data: list of (features_list,)
    """
    
    # Phase 1: Count total instances and class frequencies
    train_rdd = sc.parallelize(train_data)
    
    # Get total count and class counts
    total_count = train_rdd.count()
    class_counts = train_rdd.map(lambda x: (x[1], 1)).reduceByKey(lambda a, b: a + b).collectAsMap()
    
    # Phase 2: MapReduce for parameter estimation
    # Map: emit (attribute_value_class, 1) pairs    
    # Reduce: count occurrences
    parameter_counts = (train_rdd
                       .flatMap(map_parameters)
                       .reduceByKey(lambda a, b: a + b)
                       .collectAsMap())
    
    # Phase 3: Calculate probabilities 
    priors = calculate_priors(class_counts, total_count)
    likelihoods = calculate_likelihoods(parameter_counts, class_counts)
    
    # Phase 4: Classification
    test_rdd = sc.parallelize(test_data)
    predictions = test_rdd.map(lambda features: (features, classify_instance(features, priors, likelihoods, class_counts)))
    
    return predictions.collect()


def naive_bayes_df(spark, train_data, test_data):
    """
    DataFrame-based Naive Bayes using Spark SQL operations
    
    train_data: list of (id, features_list, class_label)  
    test_data: list of (id, features_list)
    """
    
    # Create DataFrames
    train_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("features", ArrayType(StringType()), True),
        StructField("class_label", StringType(), True)
    ])
    
    test_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("features", ArrayType(StringType()), True)
    ])
    
    train_df = spark.createDataFrame(train_data, train_schema)
    test_df = spark.createDataFrame(test_data, test_schema)
    
    # Phase 1: Calculate class priors
    total_count = train_df.count()
    class_counts_df = (train_df
                      .groupBy("class_label")
                      .agg(F.count("*").alias("class_count"))
                      .withColumn("prior", F.col("class_count") / total_count))
    
    # Phase 2: Explode features and create attribute-value-class combinations
    exploded_df = (train_df
                  .select("class_label", F.posexplode("features").alias("attr_idx", "attr_value"))
                  .withColumn("attr_key", F.concat_ws("-", F.lit("attr"), F.col("attr_idx"), F.col("attr_value"))))
    
    # Phase 3: Count attribute-value-class combinations (MapReduce equivalent)
    likelihood_counts = (exploded_df
                        .groupBy("class_label", "attr_key")
                        .agg(F.count("*").alias("attr_count")))
    
    # Join with class counts for likelihood calculation
    likelihood_df = (likelihood_counts
                    .join(class_counts_df, "class_label")
                    .withColumn("likelihood", 
                               (F.col("attr_count") + 1) / (F.col("class_count") + 2)))  # Laplace smoothing
    
    # Collect probabilities for broadcasting
    class_priors = {row['class_label']: row['prior'] for row in class_counts_df.collect()}
    likelihood_map = {}
    
    for row in likelihood_df.collect():
        class_label = row['class_label']
        attr_key = row['attr_key']
        likelihood = row['likelihood']
        
        if class_label not in likelihood_map:
            likelihood_map[class_label] = {}
        likelihood_map[class_label][attr_key] = likelihood
    
    # Broadcast the model parameters
    broadcast_priors = spark.sparkContext.broadcast(class_priors)
    broadcast_likelihoods = spark.sparkContext.broadcast(likelihood_map)
    
    # Phase 4: Classification UDF
    def classify_udf(features):
        priors = broadcast_priors.value
        likelihoods = broadcast_likelihoods.value
        
        class_scores = {}
        
        for class_label in priors:
            log_score = math.log(priors[class_label])
            
            for attr_idx, attr_value in enumerate(features):
                attr_key = f"attr-{attr_idx}-{attr_value}"
                if class_label in likelihoods and attr_key in likelihoods[class_label]:
                    log_score += math.log(likelihoods[class_label][attr_key])
                else:
                    # Default probability for unseen attributes
                    log_score += math.log(0.001)
            
            class_scores[class_label] = log_score
        
        return max(class_scores.items(), key=lambda x: x[1])[0]
    
    classify_spark_udf = F.udf(classify_udf, StringType())
    
    # Apply classification
    predictions_df = test_df.withColumn("predicted_class", classify_spark_udf(F.col("features")))
    
    return predictions_df.select("id", "predicted_class").collect()


if __name__ == "__main__":
    spark = SparkSession.builder.appName("KNN_NaiveBayes_Implementation").getOrCreate()
    sc = spark.sparkContext

    # Test data for Naive Bayes
    print("\n=== Naive Bayes Results ===")
    
    # Simple categorical data
    nb_train_rdd = [
        (["sunny", "hot", "high", "false"], "no"),
        (["sunny", "hot", "high", "true"], "no"), 
        (["overcast", "hot", "high", "false"], "yes"),
        (["rain", "mild", "high", "false"], "yes"),
        (["rain", "cool", "normal", "false"], "yes"),
        (["rain", "cool", "normal", "true"], "no"),
        (["overcast", "cool", "normal", "true"], "yes"),
        (["sunny", "mild", "high", "false"], "no"),
        (["sunny", "cool", "normal", "false"], "yes"),
        (["rain", "mild", "normal", "false"], "yes")
    ]
    
    nb_test_rdd = [
        ["sunny", "mild", "normal", "true"],
        ["overcast", "hot", "normal", "false"]
    ]
    
    nb_train_df = [
        (i, row[0], row[1]) for i, row in enumerate(nb_train_rdd)
    ]
    
    nb_test_df = [
        (i, test_case) for i, test_case in enumerate(nb_test_rdd)
    ]

    print("Naive Bayes RDD implementation:")
    nb_result_rdd = naive_bayes_rdd(sc, nb_train_rdd, nb_test_rdd)
    for result in nb_result_rdd:
        print(f"Features: {result[0]} -> Predicted: {result[1]}")

    print("\nNaive Bayes DataFrame implementation:")
    nb_result_df = naive_bayes_df(spark, nb_train_df, nb_test_df)
    for result in nb_result_df:
        print(f"ID: {result['id']} -> Predicted: {result['predicted_class']}")
    
    spark.stop()