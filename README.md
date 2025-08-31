# Spark Classification Algorithms (k-NN and Naive Bayes)

This repository contains research-oriented implementations of two classification algorithms adapted for distributed processing with Apache Spark: a MapReduce-style k-Nearest Neighbors (k-NN) and a Naive Bayes classifier. Both algorithms are implemented in two flavors: a low-level RDD-based version that follows the MapReduce steps, and a Spark DataFrame-based version using higher-level Spark SQL operations. These implementations are based on 2 papers that are in the repository

Contents

- `knn_implementations.py` — RDD and DataFrame implementations of the paper-based k-NN algorithm.
- `naive_bayes_implementation.py` — RDD and DataFrame implementations of a MapReduce-style Naive Bayes classifier.
- `knn_nursery_test.py` — Script that benchmarks k-NN implementations on the UCI Nursery dataset and reports time/memory/accuracy.
- `naive_bayes_nursery_test.py` — Script that benchmarks Naive Bayes implementations on the UCI Nursery dataset and reports time/memory/accuracy.
- `pyproject.toml` — project metadata and Python dependency pins.
- Several academic papers (PDFs) used as references for the implementations.

Requirements

- Python 3.11+
- Java (required by Spark)
- Apache Spark (the repository uses `pyspark` and was tested with Spark 4.x via the `pyspark` Python package)
- Python packages listed in `pyproject.toml`: `pandas`, `psutil`, `pyspark`, `scikit-learn`

Installation

1. Create a Python 3.11 virtual environment (recommended).
2. Install dependencies. Example (using pip):

   pip install -r requirements.txt

   or use the `pyproject.toml` with your preferred tool (poetry, pip-tools, etc.).

Notes about Spark

- The benchmark scripts initialize a local Spark session. Default memory is configured in the test scripts (e.g. `spark.driver.memory` / `spark.executor.memory`). Adjust those settings to match your machine when running benchmarks.
- The implementations collect some model parameters to the driver (matching the experimental MapReduce approach). They are not optimized for very large models/datasets.

How to run the benchmarks

- Naive Bayes benchmark (Nursery dataset):

  python3 naive_bayes_nursery_test.py

- k-NN benchmark (Nursery dataset):

  python3 knn_nursery_test.py

Each script downloads the UCI Nursery dataset at runtime and runs both the RDD and DataFrame implementations, printing execution time, memory delta (measured with `psutil`), and accuracy.

Implementation notes

- k-NN (RDD): partitions training data and computes local k-NN per partition (map), merges candidate neighbor sets (reduce), then applies majority voting. This mirrors a MapReduce paper algorithm.
- k-NN (DataFrame): cross-joins test and training data, computes distances in Spark SQL, selects k nearest per partition and globally, and performs group-based majority voting.
- Naive Bayes (RDD): counts attribute-value-class statistics with flatMap/reduceByKey, computes priors and likelihoods on the driver, then classifies test instances using log-probabilities and Laplace smoothing.
- Naive Bayes (DataFrame): uses explode/aggregation to compute likelihoods and priors, collects probabilities to driver, then applies a broadcasted UDF for classification to match the RDD behavior.

Data source

- UCI Nursery dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data
