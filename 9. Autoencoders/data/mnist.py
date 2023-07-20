import os
import numpy as np
import pandas as pd
from data.dataset import CDataSet, SetType

# =========================================================================================================================
class CMNISTDataSet(CDataSet):
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self):
    super(CMNISTDataSet, self).__init__()
   
    dTrain  = pd.read_csv(os.path.join("MLData", "mnist_train.csv"), header=None)
    nArray = dTrain.to_numpy()
    self.LoadSet(nArray[:,1:], nArray[:,0], SetType.TRAINING_SET)

    dTest   = pd.read_csv(os.path.join("MLData", "mnist_test.csv"), header=None)
    nArray = dTest.to_numpy()
    self.LoadSet(nArray[:,1:], nArray[:,0], SetType.UNKNOWN_TEST_SET)
  # --------------------------------------------------------------------------------------
# =========================================================================================================================


if __name__ == "__main__":
  oMNIST = CMNISTDataSet()
  print("[MNIST] Total Samples:%d   | Features:%d | Classes: %d" % (oMNIST.SampleCount, oMNIST.FeatureCount, oMNIST.ClassCount))
  print("[MNIST] Training:%d        |" % (oMNIST.TSSampleCount))
  print("[MNIST] Test:%d            |" % (oMNIST.USSampleCount))


