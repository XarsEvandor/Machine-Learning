import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data import CDataSet, SetType
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import datasets


# Loading a toy dataset supported by Scikit Learn
oDigits = datasets.load_digits()

# [PYTHON] Reading data from CSV into Pandas Dataframes
dMNIST  = pd.read_csv(os.path.join("MLData", "mnist_kaggle_some_rows.csv"), header=None)

# Keeps only 100 samples from the digits dataset
oDIGITSDataSet = CDataSet()
oDIGITSDataSet.LoadSet(  p_nSamples=oDigits.images[:100,...].reshape(-1, 64)  # [PYTHON] We flatten the features of the images tensor [100,8,8] to  [100,64]
                         , p_nLabels=oDigits.target[:100], p_nType=SetType.TRAINING_SET)
print("[DIGITS] Samples:%d | Features:%d | Classes: %d" % (oDIGITSDataSet.TSSampleCount, oDIGITSDataSet.FeatureCount, oDIGITSDataSet.ClassCount))

# Converts data from Pandas format to numpy array and stores them inside the MNIST dataset object
oMNISTDataSet = CDataSet()
nArray = dMNIST.to_numpy()
# The first item in each sample is the label 0-9 of the digit
oMNISTDataSet.LoadSet(  p_nSamples=nArray[:,1:]
                        , p_nLabels=nArray[:,0:], p_nType=SetType.TRAINING_SET)
print("[MNIST] Samples:%d | Features:%d | Classes: %d" % (oMNISTDataSet.TSSampleCount, oMNISTDataSet.FeatureCount, oMNISTDataSet.ClassCount))


# Select from DIGITS and MNIST 100 sample sets
nTrainingSamples = oDIGITSDataSet.TSSamples


# // Hyperparameters \\
COMPONENTS_FROM = 2
COMPONENTS_TO   = 50



oPCAModels = []
for nComponents in range(COMPONENTS_FROM, COMPONENTS_TO + 1):
  nModelIndex = nComponents - COMPONENTS_FROM
  print("-"*25, "PCA model %d with %d components" % (nModelIndex + 1, nComponents), "-"*25)
  oPCA = PCA(n_components=nComponents)
  oPCA.fit(nTrainingSamples)
  nRatioExplainedVariance = np.sum(oPCA.explained_variance_ratio_)
  print("Ratio of total explained variance of %d components:%.4f" % (nComponents, nRatioExplainedVariance) )

  # Keep the model packed with hyperparameter and metric
  oPCAModels.append([oPCA, nComponents, nRatioExplainedVariance])

# Comparison of different PCA models for the same dataset
nX = [oRec[1] for oRec in oPCAModels] # [PYTHON] This is called list comprehension. The iteration creates the items of the list as describes before the for statement.
nY = [oRec[2] for oRec in oPCAModels] 

# X axis the components, Y axis the ratio of total explained variance
plt.plot(nX, nY)
plt.show()




nModelIndex = 24
oPCA, nComponents, nRatioExplainedVariance = oPCAModels[nModelIndex] # [PYTHON] Unpack to three variables
print("="*60)
print("Analysis for PCA Model %d" % (nModelIndex + 1))
print("|__ Ratio of total explained variance for %d components:%.4f" % (nComponents, nRatioExplainedVariance) )
for nIndex in range(0, nComponents):
  print("     |__Component %.2d variance: %.4f , eigenvalue:%.4f" % (nIndex + 1, oPCA.explained_variance_ratio_[nIndex], oPCA.singular_values_[nIndex]))
        

