import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from data.digits import CDIGITSDataSet
from data.mnist import CMNISTDataSet

from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lib.evaluation import CEvaluator

# --------------------------------------------------------------------------------------
def EvaluateKMeans(k, kmeans, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(data, estimator[-1].labels_,
                                 metric="euclidean", sample_size=300,)
    ]

    # Show the results
    formatter_result = ("%d\t" % k + "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}"
                        "\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}")
    print(formatter_result.format(*results))
# --------------------------------------------------------------------------------------




oDataSet = CMNISTDataSet() 

# _____// Hyperparameters \\_____


# ... Data Hyperparameters ...
SAMPLE_COUNT = 1000 # How many samples to use from the available population in the dataset

# ... Dimensionality Reduction Hyperparameters ...
COMPONENTS = 2

# ... Dimensionality Reduction Hyperparameters ...
PERPLEXITY      = 100.0
LEARNING_RATE   = 1000.0
EPOCHS          = 1000

# ... Clustering Hyperparameters ...
NUM_RUNS_RANDOM_CENTROIDS = 4
RANDOM_SEED               = 2021
CLUSTER_COUNT_FROM        = 4
CLUSTER_COUNT_TO          = 30


data      = oDataSet.TSSamples[:SAMPLE_COUNT,:]
labels    = oDataSet.TSLabels[:SAMPLE_COUNT]

print("Reducing dimensionsality to 2 using t-SNE")
oTSNE = TSNE(n_components=COMPONENTS ,perplexity=PERPLEXITY, n_iter=EPOCHS)


#reduced_data  = PCA(n_components=COMPONENTS).fit_transform(data)
reduced_data  = oTSNE.fit_transform(data)
reduced_data = reduced_data.astype(np.float64)


print(82 * '_')
print('init\t\ttime\tinertia\thomogeneity\tcompleteness\tv_measure\tARI\tAMI\tsilhouette')

for k in range(CLUSTER_COUNT_FROM, CLUSTER_COUNT_TO):
  oKMeansModel = KMeans(init="k-means++", n_clusters=k, n_init=NUM_RUNS_RANDOM_CENTROIDS)
  EvaluateKMeans(k, kmeans=oKMeansModel, name="k-means++", data=data, labels=labels)

print(82 * '_')
