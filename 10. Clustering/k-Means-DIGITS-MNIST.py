import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from data.digits import CDIGITSDataSet
from data.mnist import CMNISTDataSet
from rx.voronoi import CVoronoi2DPlot

from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
COMPONENTS = 2
IS_LINEAR_DIM_REDUCTION = False

# ... Dimensionality Reduction Hyperparameters ...
PERPLEXITY      = 100.0
LEARNING_RATE   = 1000.0
EPOCHS          = 1000
GRADIENT_CALCULATION_ALGORITHM = "barnes_hut" #Fast

if IS_LINEAR_DIM_REDUCTION:
  sDimReductionMethod = "PCA"
  nReducedSamples  = PCA(n_components=COMPONENTS).fit_transform(nSamples)
else:
  sDimReductionMethod = "t-SNE"
  oTSNE = TSNE( n_components=COMPONENTS
                ,perplexity=PERPLEXITY, n_iter=EPOCHS
                ,method=GRADIENT_CALCULATION_ALGORITHM
                ,verbose=2
                )
  nReducedSamples  = oTSNE.fit_transform(nSamples)

# _____// Clustering Hyperparameters \\_____
NUM_RUNS_RANDOM_CENTROIDS = 4
RANDOM_SEED               = 2021
CLUSTER_COUNT_K           = 12

nReducedSamples = nReducedSamples.astype(np.float64)
oClusteringModel = KMeans(init="k-means++", n_clusters=CLUSTER_COUNT_K, n_init=NUM_RUNS_RANDOM_CENTROIDS, verbose=2)
oClusteringModel.fit(nReducedSamples)


oVoronoi = CVoronoi2DPlot("K-means clustering on %s dataset (%s reduced data)\n" 
                          "Centroids are marked with white cross" % (sDataName, sDimReductionMethod)
                          ,nReducedSamples, nLabels, p_nGroundTruthClusterCount=10)
oVoronoi.ShowForKMeans(oClusteringModel)


