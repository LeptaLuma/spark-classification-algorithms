from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import time
from pyspark.sql import SparkSession
from knn_implementations import knn_rdd, knn_df
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Initialize Spark
spark = SparkSession.builder \
    .appName("kNN_Benchmark") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .getOrCreate()
sc = spark.sparkContext

def load_nursery_data():
    """Load and prepare nursery dataset"""
    # Do URL download
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"
    
    columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']
    
    df = pd.read_csv(url, names=columns)
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in df.columns:
        df[col] = le.fit_transform(df[col])
    
    X = df.drop('class', axis=1).values
    y = df['class'].values
    
    return X, y

def test_nursery_knn(k=5):
    """Test kNN on nursery dataset"""
    print("Loading nursery dataset...")
    X, y = load_nursery_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {X.shape[1]}")
    
    # Prepare data for kNN functions
    train_rdd = [(X_train[i].tolist(), str(y_train[i])) for i in range(len(X_train))]
    test_rdd = [(X_test[i].tolist(), None) for i in range(len(X_test))]
    
    train_data = [(i, X_train[i].tolist(), str(y_train[i])) for i in range(len(X_train))]
    test_data = [(i, X_test[i].tolist()) for i in range(len(X_test))]
    
    n_partitions = 10

    # Test RDD
    print("\nTesting RDD implementation...")
    start = time.time()
    rdd_result = knn_rdd(sc, train_rdd, test_rdd, k, n_partitions)
    rdd_time = time.time() - start
    print(f"RDD time: {rdd_time:.2f}s")
    
    # RDD accuracy & report
    # rdd_result: list of (id_test, predicted_class)
    rdd_preds = [pred for _, pred in sorted(rdd_result, key=lambda x: x[0])]
    true_labels = [str(lbl) for lbl in y_test]
    acc_rdd = accuracy_score(true_labels, rdd_preds)
    print(f"RDD Accuracy: {acc_rdd:.4f}")

    # Test DataFrame
    print("Testing DataFrame implementation...")
    start = time.time()
    df_result = knn_df(spark, train_data, test_data, k, n_partitions)
    df_time = time.time() - start
    print(f"DataFrame time: {df_time:.2f}s")

    # DF accuracy & report
    # df_result: Spark DataFrame with columns id_test, predicted_class
    df_list = df_result.collect()
    df_preds = [row.predicted_class for row in sorted(df_list, key=lambda r: r.id_test)]
    acc_df = accuracy_score(true_labels, df_preds)
    print(f"DF Accuracy: {acc_df:.4f}")
    
    print(f"\nSpeedup: {rdd_time/df_time:.2f}x")

if __name__ == "__main__":
    test_nursery_knn()
    spark.stop()