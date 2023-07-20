import numpy as np
from enum import Enum


# =========================================================================================================================
class SetType(Enum):
  TRAINING_SET      = 1
  VALIDATION_SET    = 2
  UNKNOWN_TEST_SET  = 3
# =========================================================================================================================





# =========================================================================================================================
class CDataSet(object):
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self):
    # ................................................................
    # // Fields \\
    self.Samples            = None
    self.Labels             = None
    self.SampleCount        = 0
    self.FeatureCount       = None
    self.ClassCount         = None

    self.TSSamples      = None
    self.TSLabels       = None
    self.TSSampleCount  = 0

    self.VSSamples      = None
    self.VSLabels       = None
    self.VSSampleCount  = 0

    self.USSamples      = None
    self.USLabels       = None
    self.USSampleCount  = 0
    # ................................................................
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
  def LoadSet(self, p_nSamples, p_nLabels, p_nType):
    if p_nType == SetType.TRAINING_SET:
      self.TSSamples = p_nSamples
      self.TSLabels  = p_nLabels
      self.TSSampleCount = self.TSSamples.shape[0]
      if self.FeatureCount is None:
        self.FeatureCount = self.TSSamples.shape[1]
        self.ClassCount = len(np.unique(self.TSLabels))
    elif p_nType == SetType.VALIDATION_SET:
      self.VSSamples = p_nSamples
      self.VSLabels = p_nLabels
      self.VSSampleCount = self.VSSamples.shape[0]
    elif p_nType == SetType.UNKNOWN_TEST_SET:
      self.USSamples = p_nSamples
      self.USLabels = p_nLabels
      self.USSampleCount = self.USSamples.shape[0]

    self.SampleCount = self.TSSampleCount + self.VSSampleCount + self.USSampleCount
  # --------------------------------------------------------------------------------------
  def Split(self, p_nTrainingPercentage):
    self.TSSamples, self.VSSamples, self.TSLabels, self.VSLabels = train_test_split(
                                                              self.Samples, self.Labels
                                                            , test_size=1-p_nTrainingPercentage, random_state=2021)
        
    self.TSSampleCount = self.TSSamples.shape[0]
    self.VSSampleCount = self.VSSamples.shape[0]
  # --------------------------------------------------------------------------------------
# =========================================================================================================================
