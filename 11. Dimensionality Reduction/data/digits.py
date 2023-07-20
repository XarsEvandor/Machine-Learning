from sklearn import datasets
from data.dataset import CDataSet, SetType

# =========================================================================================================================
class CDIGITSDataSet(CDataSet):
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self):
    super(CDIGITSDataSet, self).__init__()
    oDigits = datasets.load_digits()
    self.LoadSet(oDigits.images.reshape(-1, 64), oDigits.target, SetType.TRAINING_SET)
  # --------------------------------------------------------------------------------------
# =========================================================================================================================


if __name__ == "__main__":
  oDIGITS = CDIGITSDataSet()
  print("[DIGITS] Samples:%d | Features:%d | Classes: %d" % (oDIGITS.TSSampleCount, oDIGITS.FeatureCount, oDIGITS.ClassCount))





