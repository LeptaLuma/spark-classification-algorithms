from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import time
import psutil
from pyspark.sql import SparkSession
from naive_bayes_implementation import naive_bayes_rdd, naive_bayes_df
from sklearn.metrics import accuracy_score

# Initialize Spark
spark = SparkSession.builder \
    .appName("NaiveBayes_Benchmark") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()
sc = spark.sparkContext

def get_memory_mb():
    """Get current memory usage in MB"""
    return psutil.Process().memory_info().rss / (1024 * 1024)

def load_nursery_data():
    """Load and prepare nursery dataset"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"
    columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']
    
    df = pd.read_csv(url, names=columns)
    
    # For Naive Bayes, keep categorical data as strings instead of encoding to numbers
    # This makes the algorithm more interpretable and follows the paper's approach
    X = df.drop('class', axis=1).values
    y = df['class'].values
    
    return X, y

def benchmark_function(name, func, *args):
    """Simple benchmark with memory tracking"""
    print(f"\n--- {name} ---")
    
    # Measure before
    mem_before = get_memory_mb()
    start_time = time.time()
    
    # Run function
    result = func(*args)
    
    # Force execution for Spark DataFrames
    if hasattr(result, 'collect'):
        result = result.collect()
    
    # Measure after
    end_time = time.time()
    mem_after = get_memory_mb()
    
    execution_time = end_time - start_time
    memory_used = mem_after - mem_before
    
    print(f"Time: {execution_time:.2f}s")
    print(f"Memory: {memory_used:.1f} MB")
    
    return {
        'result': result,
        'time': execution_time,
        'memory': memory_used
    }

def test_nursery_naive_bayes():
    """Simple test comparison for Naive Bayes"""
    print("Loading dataset...")
    X, y = load_nursery_data()
    
    # Split data (70/30 as mentioned in the paper)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {X.shape[1]}")
    
    # Show class distribution
    unique_classes, counts = pd.Series(y_train).value_counts().index, pd.Series(y_train).value_counts().values
    print("Class distribution in training set:")
    for cls, count in zip(unique_classes, counts):
        print(f"  {cls}: {count} ({count/len(y_train)*100:.1f}%)")
    
    # Prepare data for RDD implementation
    # Convert features to list of strings (categorical data)
    train_rdd = [(X_train[i].tolist(), str(y_train[i])) for i in range(len(X_train))]
    test_rdd = [X_test[i].tolist() for i in range(len(X_test))]
    
    # Prepare data for DataFrame implementation
    train_data = [(i, X_train[i].tolist(), str(y_train[i])) for i in range(len(X_train))]
    test_data = [(i, X_test[i].tolist()) for i in range(len(X_test))]
    
    # Test RDD Implementation
    rdd_metrics = benchmark_function(
        "Naive Bayes RDD Implementation",
        naive_bayes_rdd, sc, train_rdd, test_rdd
    )
    
    # Calculate accuracy for RDD
    rdd_preds = [pred for _, pred in rdd_metrics['result']]
    true_labels = [str(lbl) for lbl in y_test]
    rdd_accuracy = accuracy_score(true_labels, rdd_preds)
    print(f"Accuracy: {rdd_accuracy:.4f}")
    
    # Test DataFrame Implementation
    df_metrics = benchmark_function(
        "Naive Bayes DataFrame Implementation", 
        naive_bayes_df, spark, train_data, test_data
    )
    
    # Calculate accuracy for DataFrame
    df_preds = [row['predicted_class'] for row in sorted(df_metrics['result'], key=lambda r: r['id'])]
    df_accuracy = accuracy_score(true_labels, df_preds)
    print(f"Accuracy: {df_accuracy:.4f}")
    
    # Compare results
    print(f"\n{'='*50}")
    print("COMPARISON")
    print(f"{'='*50}")
    
    time_speedup = rdd_metrics['time'] / df_metrics['time']
    memory_ratio = rdd_metrics['memory'] / max(abs(df_metrics['memory']), 1)
    
    print(f"Time - RDD: {rdd_metrics['time']:.2f}s, DF: {df_metrics['time']:.2f}s")
    print(f"Speedup: {time_speedup:.2f}x ({'RDD faster' if time_speedup < 1 else 'DF faster'})")
    
    print(f"Memory - RDD: {rdd_metrics['memory']:.1f}MB, DF: {df_metrics['memory']:.1f}MB")
    print(f"Memory ratio: {memory_ratio:.2f}x")
    
    print(f"Accuracy - RDD: {rdd_accuracy:.4f}, DF: {df_accuracy:.4f}")
    print(f"Accuracy diff: {abs(rdd_accuracy - df_accuracy):.6f}")

if __name__ == "__main__":
    test_nursery_naive_bayes()
    spark.stop()