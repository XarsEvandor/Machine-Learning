import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from data.digits import CDIGITSDataSet
from data.mnist import CMNISTDataSet

from rx.voronoi import CVoronoi2DPlot
from rx.scatter import CMultiScatterPlot
from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# _____// Data Hyperparameters \\_____
IS_MNIST = True
SAMPLE_COUNT = 1500  # How many samples to use from the available population in the dataset

if IS_MNIST:
  oMNIST  = CMNISTDataSet()
  sDataName  = "MNIST"
  nSamples   = oMNIST.TSSamples[:SAMPLE_COUNT,:]
  nLabels    = oMNIST.TSLabels[:SAMPLE_COUNT]
else:
  sDataName = "DIGITS"
  oDIGITS = CDIGITSDataSet() 
  nSamples   = oDIGITS.TSSamples[:SAMPLE_COUNT,:]
  nLabels    = oDIGITS.TSLabels[:SAMPLE_COUNT]


# _____// Dimensionality Reduction Hyperparameters \\_____
IS_LINEAR_DIM_REDUCTION = False
COMPONENTS = 2
# ....... t-SNE .......
PERPLEXITY      = 100.0
LEARNING_RATE   = 1000.0
EPOCHS          = 1000

#nSamples = StandardScaler().fit_transform(nSamples)

nReducedSamplesPCA  = PCA(n_components=COMPONENTS).fit_transform(nSamples)
if IS_LINEAR_DIM_REDUCTION:
  sDimReductionMethod = "PCA"
  nReducedSamples  = nReducedSamplesPCA
else:
  sDimReductionMethod = "t-SNE"
  oTSNE = TSNE(n_components=COMPONENTS,perplexity=PERPLEXITY, n_iter=EPOCHS)
  nReducedSamples  = oTSNE.fit_transform(nSamples)
print("Reduced dimensionality to 2 using %s" % sDimReductionMethod)

nReducedSamples = nReducedSamples.astype(np.float64)

# _____// Clustering Hyperparameters \\_____
# ...... DBSCAN ......
MAX_DISTANCE_OF_NEIGHBORS = 2
NEIGHBORHOOD_SIZE         = 5
# ...... k-MEANS ......
KAPPA                     = 12
NUM_RUNS_RANDOM_CENTROIDS = 4
RANDOM_SEED               = 2021

# k-Means learns k centroids from data, so a cluster label 
oClusteringModel = KMeans(init="k-means++", n_clusters=KAPPA, n_init=NUM_RUNS_RANDOM_CENTROIDS, verbose=2)
oClusteringModel.fit(nReducedSamples)

# Visualize clustering
oVoronoi = CVoronoi2DPlot("K-means clustering on %s dataset (%s reduced data)\n" 
                          "Centroids are marked with white cross" % (sDataName, sDimReductionMethod)
                          ,nReducedSamples, nLabels, p_nGroundTruthClusterCount=10)
oVoronoi.ShowForKMeans(oClusteringModel)


# DBScan estimates number of clusters and assigns label to data. It does not work on uknown samples.
print("Training DBSCAN")
nReducedSamplesPCA = nReducedSamplesPCA.astype(np.float64)
oDBScan = DBSCAN(eps=MAX_DISTANCE_OF_NEIGHBORS, min_samples=NEIGHBORHOOD_SIZE)
nClusterLabels = oDBScan.fit_predict(nReducedSamplesPCA)

#nClusterLabels = oDBScan.labels_
nNumClusters = len(set(nClusterLabels)) - (1 if -1 in nClusterLabels else 0)
print("number of clusters in pca-DBSCAN: ", nNumClusters)


# Visualize clustering
oPlot = CMultiScatterPlot("DBSCAN clustering on %s dataset (%s reduced data)" % (sDataName,sDimReductionMethod))
oPlot.AddData("DBSCAN", nReducedSamples, nClusterLabels)
oPlot.Show(0, "Component1", "Component2")