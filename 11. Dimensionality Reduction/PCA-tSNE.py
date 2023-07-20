import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from data.digits import CDIGITSDataSet
from data.mnist import CMNISTDataSet

# =========================================================================================================================
class CUnsupervisedDimReduction(object):
  # --------------------------------------------------------------------------------------
  def __init__(self, p_sName, p_nSamples):
    self.Name       = p_sName
    self.Samples    = p_nSamples
    self.PCAModels  = []
    self.TSNEModels = []
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


# _____// Hyperparameters \\_____
# ... Data Hyperparameters ...
SAMPLE_COUNT = 100 # How many samples to use from the available population in the dataset

# ... Dimensionality Reduction Hyperparameters ...
COMPONENTS_FROM = 2
COMPONENTS_TO   = 40
PERPLEXITY      = 100.0
LEARNING_RATE   = 200.0
EPOCHS          = 1000
GRADIENT_CALCULATION_ALGORITHM = "exact" #Slow but for higher number of components
#GRADIENT_CALCULATION_ALGORITHM = "barnes_hut" #Fast


oDRModels = []
oDRModels.append(CUnsupervisedDimReduction("DIGITS", oDIGITS.TSSamples[:SAMPLE_COUNT,:]))
oDRModels.append(CUnsupervisedDimReduction("MNIST" , oMNIST.TSSamples[:SAMPLE_COUNT,:]))


# Creating PCA Models
for oDRM in oDRModels:
  for nComponents in range(COMPONENTS_FROM, COMPONENTS_TO + 1):
    print("-"*25, "PCA on %s with %d Components" % (oDRM.Name, nComponents), "-"*25)
    oPCA = PCA(n_components=nComponents)
    oPCA.fit(oDRM.Samples)
    oDRM.AddPCAModel(oPCA)

# Comparison of different PCA models
plt.title("Principle Component Analysics (PCA)")
plt.xlabel('Components')
plt.ylabel('Ratio of Explained Variance')

nX1, nY1 = oDRModels[0].GetPCAPlotSerie()
plt.plot(nX1, nY1, "-b", label=oDRModels[0].Name)

nX2, nY2 = oDRModels[1].GetPCAPlotSerie()
plt.plot(nX2, nY2, "-r", label=oDRModels[1].Name)

plt.legend(loc="upper left")
plt.xlim(0.0, COMPONENTS_TO + 5)
plt.show()



# Learning t-SNE Embeddings
for oDRM in oDRModels:
  for nComponents in range(COMPONENTS_FROM, COMPONENTS_TO + 1):
    print("-"*25, "t-SNE on %s with %d Components" % (oDRM.Name, nComponents), "-"*25)
    oTSNE = TSNE( n_components=nComponents
                 ,perplexity=PERPLEXITY, n_iter=EPOCHS
                 ,method=GRADIENT_CALCULATION_ALGORITHM
                 )
    oTSNE.fit(oDRM.Samples)
    oDRM.AddTSNEModel(oTSNE, nComponents)

# Comparison of different t-SNE models
plt.title("t-Distributed Stochastic Neighbor Embedding (t-SNE)")
plt.xlabel('Components')
plt.ylabel('Kullback–Leibler Divergence (Cost)')
nX1, nY1 = oDRModels[0].GetTSNEPlotSerie()
plt.plot(nX1, nY1, "-b", label=oDRModels[0].Name)

nX2, nY2 = oDRModels[1].GetTSNEPlotSerie()
plt.plot(nX2, nY2, "-r", label=oDRModels[1].Name)

plt.legend(loc="upper left")
plt.xlim(0.0, COMPONENTS_TO + 5)
plt.show()












