from sklearn.model_selection import train_test_split    # import a standalone procedure function from the pacakge


# =========================================================================================================================
class CCustomDataSet(object):
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self):
    # ................................................................
    # // Fields \\
    self.Samples            = None
    self.Labels             = None
    self.SampleCount        = None
    self.FeatureCount       = None
    self.ClassCount         = None

    self.TSSamples          = None
    self.TSLabels           = None
    self.TSSampleCount      = 0

    self.VSSamples          = None
    self.VSLabels           = None
    self.VSSampleCount      = 0
    # ................................................................
  # --------------------------------------------------------------------------------------
  def Split(self, p_nTrainingPercentage):
    self.TSSamples, self.VSSamples, self.TSLabels, self.VSLabels = train_test_split(
                                                              self.Samples, self.Labels
                                                            , test_size=1-p_nTrainingPercentage, random_state=2021)
        
    self.TSSampleCount = self.TSSamples.shape[0]
    self.VSSampleCount = self.VSSamples.shape[0]

  # --------------------------------------------------------------------------------------
# =========================================================================================================================
