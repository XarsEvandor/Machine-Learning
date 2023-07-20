import numpy as np                      # use the package (a.k.a. namespace) with the alias "np"
from sklearn import datasets            # import as single object/subpackage from the package
from sklearn import preprocessing   
from sklearn.model_selection import train_test_split    # import a standalone procedure function from the pacakge
from lib.utils import RandomSeed                      # import a standalone procedure function from the pacakge
from rx.visualization import CPlot
from lib.data.dataset import CDataSet

# =========================================================================================================================
class CMyDataset(object):
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self, p_nSampleCount=1000, p_nFeatureCount=2, p_nClassCount=2, p_nClassSeperability=1.0, p_nClustersPerClass=1):
    # ................................................................
    # // Fields \\
    self.Samples            = None
    self.Labels             = None
    self.SampleCount        = p_nSampleCount
    self.FeatureCount       = p_nFeatureCount
    self.ClassCount         = p_nClassCount
    self.ClassSeperability  = p_nClassSeperability
    self.ClustersPerClass   = p_nClustersPerClass

    self.TSSamples = None
    self.TSLabels  = None
    self.TSSampleCount = 0

    self.VSSamples = None
    self.VSLabels  = None
    self.VSSampleCount = 0
    # ................................................................

    RandomSeed(2021)

    
    self.CreateData() # virtual

  # --------------------------------------------------------------------------------------
  def CreateData(self):
    self.Samples, self.Labels = datasets.make_classification(
        n_features=self.FeatureCount,
        n_classes=self.ClassCount,
        n_samples=self.SampleCount,
        n_informative=self.FeatureCount,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=self.ClassSeperability
    )


    
  # --------------------------------------------------------------------------------------
  # Method 
  def DebugPrint(self):
    print("Shape of sample tensor", self.Samples.shape)
    print('.'*80)

    print("Datatype of sample tensor before convertion: %s" % str(self.Samples.dtype))
    # Convert the data to 32bit floating point numbers (default for faster computations)
    self.Samples = np.asarray(self.Samples, dtype=np.float32)
    print("Datatype of sample tensor after convertion: %s" % str(self.Samples.dtype))
    print('.'*80)

    # Classification into 2 classes == Binary classification
    print("Class labels")
    print(self.Labels)
    print('.'*80)
  # --------------------------------------------------------------------------------------
  def Split(self, p_nTrainingPercentage):
    self.TSSamples, self.VSSamples, self.TSLabels, self.VSLabels = train_test_split(
                                                              self.Samples, self.Labels
                                                            , test_size=1-p_nTrainingPercentage, random_state=2021)
        
    self.TSSampleCount = self.TSSamples.shape[0]
    self.VSSampleCount = self.VSSamples.shape[0]

  # --------------------------------------------------------------------------------------
# =========================================================================================================================




# =========================================================================================================================
class CMyDataset2D(CMyDataset):
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self, p_nSampleCount=1000, p_nFeatureCount=2, p_nClassCount=2, p_nClassSeperability=1.0, p_nClustersPerClass=1):
    super(CMyDataset2D, self).__init__(p_nSampleCount, p_nFeatureCount, p_nClassCount, p_nClassSeperability, p_nClustersPerClass)
  # --------------------------------------------------------------------------------------
  def CreateData(self):
    self.Samples, self.Labels = datasets.make_classification(
        n_features=self.FeatureCount,
        n_classes=self.ClassCount,
        n_samples=self.SampleCount,
        n_redundant=0,
        n_clusters_per_class=self.ClustersPerClass,
        class_sep=self.ClassSeperability
    )
  # --------------------------------------------------------------------------------------
# =========================================================================================================================








# This checks if the current python file is executing as the programs main method
if __name__ == "__main__":
  CLASS_COUNT = 4
  bIsMinmaxScaled = False;

  if CLASS_COUNT == 4:
    oDataset = CMyDataset(p_nSampleCount=5000, p_nFeatureCount=8, p_nClassCount=CLASS_COUNT, p_nClassSeperability=0.8)
    oDataset.DebugPrint()
    oDataset.Split(0.8)

    # Scale the features to 0 .. 1
    if bIsMinmaxScaled:
        oScaler = preprocessing.MinMaxScaler().fit(oDataset.Samples)
        oDataset.Samples = oScaler.transform(oDataset.Samples)
    

    for nIndex in range(oDataset.FeatureCount):
      if nIndex < (oDataset.FeatureCount - 1):
        oPlot = CPlot("Dataset", oDataset.Samples[:,nIndex:nIndex + 2], oDataset.Labels)
        oPlot.Show(bIsMinmaxScaled, p_nStartFeatureIndex=nIndex)
  else:
    oDataset = CMyDataset2D()
    oDataset.DebugPrint()
    oDataset.Split(0.8)

    # Scale the features to 0 .. 1
    if bIsMinmaxScaled:
        oScaler = preprocessing.MinMaxScaler().fit(oDataset.Samples)
        oDataset.Samples = oScaler.transform(oDataset.Samples)
    
    oPlot = CPlot("Dataset", oDataset.Samples[:,0:2], oDataset.Labels)
    oPlot.Show(bIsMinmaxScaled)

    oPlot = CPlot("Training Set", oDataset.TSSamples[:,0:2], oDataset.TSLabels)
    oPlot.Show(bIsMinmaxScaled)

    oPlot = CPlot("Validation Set", oDataset.VSSamples[:,0:2], oDataset.VSLabels)
    oPlot.Show(bIsMinmaxScaled)