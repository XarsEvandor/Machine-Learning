import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data.digits import CDIGITSDataSet
from data.mnist import CMNISTDataSet

# =========================================================================================================================
class CUnsupervisedDimReduction(object):
  # --------------------------------------------------------------------------------------
  def __init__(self, p_sName, p_nSamples):
    self.Name       = p_sName
    self.Samples    = p_nSamples
    self.PCAModels  = []
  # --------------------------------------------------------------------------------------
  def AddPCAModel(self, p_oPCAModel):
    self.PCAModels.append([p_oPCAModel, p_oPCAModel.n_components_, np.sum(p_oPCAModel.explained_variance_ratio_)])
  # --------------------------------------------------------------------------------------
  def GetPlotSerie(self):
    nX = [oRec[1] for oRec in self.PCAModels]
    nY = [oRec[2] for oRec in self.PCAModels] 
    return nX, nY
  # --------------------------------------------------------------------------------------
# =========================================================================================================================

oDIGITS = CDIGITSDataSet() 
oMNIST  = CMNISTDataSet()


# _____// Hyperparameters \\_____
# ... Data Hyperparameters ...
SAMPLE_COUNT = 1000 # How many samples to use from the available population in the dataset
# ... Dimensionality Reduction Hyperparameters ...
COMPONENTS_FROM = 2
COMPONENTS_TO   = 50

oDRModels = []
oDRModels.append(CUnsupervisedDimReduction("DIGITS", oDIGITS.TSSamples[:SAMPLE_COUNT,:]))
oDRModels.append(CUnsupervisedDimReduction("MNIST" , oMNIST.TSSamples[:SAMPLE_COUNT,:]))

for oDRM in oDRModels:
  for nComponents in range(COMPONENTS_FROM, COMPONENTS_TO + 1):
    print("-"*25, "PCA on %s with %d Components" % (oDRM.Name, nComponents), "-"*25)
    oPCA = PCA(n_components=nComponents)
    oPCA.fit(oDRM.Samples)
    oDRM.AddPCAModel(oPCA)


nX1, nY1 = oDRModels[0].GetPlotSerie()
plt.plot(nX1, nY1, "-b", label=oDRModels[0].Name)

nX2, nY2 = oDRModels[1].GetPlotSerie()
plt.plot(nX2, nY2, "-r", label=oDRModels[1].Name)

plt.legend(loc="upper left")
plt.xlim(0.0, COMPONENTS_TO + 5)
plt.show()
