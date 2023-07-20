import os
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from data.digits import CDIGITSDataSet
from data.mnist import CMNISTDataSet

import matplotlib.pyplot as plt
from rx.scatter import CMultiScatterPlot


# =========================================================================================================================
class CUnsupervisedEmbeddings(object):
  # --------------------------------------------------------------------------------------
  def __init__(self, p_sName, p_nSamples, p_nLabels):
    self.Name         = p_sName
    self.Samples      = p_nSamples
    self.Labels       = p_nLabels
    self.ClusterCount = len(np.unique(self.Labels))
    self.PCAModels  = []
    self.TSNEModels = []
    self.EmbeddingsLinear    = None
    self.EmbeddingsNonLinear = None
  # --------------------------------------------------------------------------------------
  def AddPCAModel(self, p_oPCAModel):
    self.PCAModels.append([p_oPCAModel, p_oPCAModel.n_components_, np.sum(p_oPCAModel.explained_variance_ratio_)])
  # --------------------------------------------------------------------------------------
  def AddTSNEModel(self, p_oTSNEModel, p_nComponents):
    self.TSNEModels.append([p_oTSNEModel, p_nComponents, p_oTSNEModel.kl_divergence_])
  # --------------------------------------------------------------------------------------
  def GetPCAPlotSerie(self):
    nX = [oRec[1] for oRec in self.PCAModels]
    nY = [oRec[2] for oRec in self.PCAModels] 
    return nX, nY
  # --------------------------------------------------------------------------------------
  def GetTSNEPlotSerie(self):
    nX = [oRec[1] for oRec in self.TSNEModels]
    nY = [oRec[2] for oRec in self.TSNEModels] 
    return nX, nY
  # --------------------------------------------------------------------------------------
# =========================================================================================================================

oDIGITS = CDIGITSDataSet() 
oMNIST  = CMNISTDataSet()



# Min-max scale the samples of DIGITS
oDIGITS.Samples = MinMaxScaler().fit_transform(oDIGITS.TSSamples)

# Min-max scale the samples of MNIST
oMNIST.Samples = MinMaxScaler().fit_transform(oMNIST.TSSamples)



# _____// Hyperparameters \\_____
# ... Data Hyperparameters ...
#SAMPLE_COUNT = 100 # t-SNE fails with few samples (needs more to learn)
#SAMPLE_COUNT = 200
SAMPLE_COUNT = 1700  

# ... Dimensionality Reduction Hyperparameters ...
COMPONENTS      = 2
PERPLEXITY      = 100.0
LEARNING_RATE   = 1000.0
EPOCHS          = 1000
#GRADIENT_CALCULATION_ALGORITHM = "exact" #Slow but for higher number of components
GRADIENT_CALCULATION_ALGORITHM = "barnes_hut" #Fast


oDRModels = []
oDRModels.append(CUnsupervisedEmbeddings("DIGITS", oDIGITS.TSSamples[:SAMPLE_COUNT,:], oDIGITS.TSLabels[:SAMPLE_COUNT]))
oDRModels.append(CUnsupervisedEmbeddings("MNIST" , oMNIST.TSSamples[:SAMPLE_COUNT,:], oMNIST.TSLabels[:SAMPLE_COUNT]))

# Creating PCA Models
for oDRM in oDRModels:
    print("-"*25, "PCA on %s with %d Components" % (oDRM.Name, COMPONENTS), "-"*25)
    oPCA = PCA(n_components=COMPONENTS)
    oPCA.fit(oDRM.Samples)
    oDRM.EmbeddingsLinear = oPCA.transform(oDRM.Samples)
    oDRM.AddPCAModel(oPCA)


    print("-"*25, "t-SNE on %s with %d Components" % (oDRM.Name, COMPONENTS), "-"*25)
    oTSNE = TSNE( n_components=COMPONENTS
                 ,perplexity=PERPLEXITY, n_iter=EPOCHS
                 ,method=GRADIENT_CALCULATION_ALGORITHM
                 ,verbose=2
                 )
    oDRM.EmbeddingsNonLinear = oTSNE.fit_transform(oDRM.Samples)
    oDRM.AddTSNEModel(oTSNE, COMPONENTS)




oPlot = CMultiScatterPlot("PCA vs t-SNE on DIGITS and MNIST")
for oDRM in oDRModels:
  oPlot.AddData("PCA %s" % oDRM.Name  , oDRM.EmbeddingsLinear   , oDRM.Labels)
  oPlot.AddData("t-SNE %s" % oDRM.Name, oDRM.EmbeddingsNonLinear, oDRM.Labels)

for nIndex in range(0, 4):
  oPlot.Show(nIndex, "Component1", "Component2")



oPlot = CMultiScatterPlot("PCA vs t-SNE on DIGITS and MNIST (No Ground Truth)")
for oDRM in oDRModels:
  oPlot.AddData("PCA %s" % oDRM.Name  , oDRM.EmbeddingsLinear   )
  oPlot.AddData("t-SNE %s" % oDRM.Name, oDRM.EmbeddingsNonLinear)

for nIndex in range(0, 4):
  oPlot.Show(nIndex, "Component1", "Component2")





