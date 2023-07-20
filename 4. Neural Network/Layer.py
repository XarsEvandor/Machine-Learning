import numpy as np
from  Neuron import CNeuron


# ====================================================================================================
class CNeuralLayer(list):
  # --------------------------------------------------------------------------------------
  def __init__(self, p_nNeuronCount, p_nNeuronInputFeatures):
    # ................................................................
    self.NeuronCount    = p_nNeuronCount
    self.InputFeatures  = p_nNeuronInputFeatures
    # ................................................................
    self.Create()
  # --------------------------------------------------------------------------------------    
  def __call__(self, p_oInput):
      return self.Recall(p_oInput)
  # --------------------------------------------------------------------------------------
  def Create(self):
    for nIndex in range(0, self.NeuronCount):
      oNeuron = CNeuron(self.InputFeatures, p_sActivationFunction='linear')
      self.append(oNeuron)
  # --------------------------------------------------------------------------------------
  def Recall(self, p_oInput):
    vLayerActivation = []
    for oNeuron in self:
      nNeuronActivation = oNeuron.Recall(p_oInput)
      vLayerActivation.append(nNeuronActivation)
    
    vLayerActivation = np.asarray(vLayerActivation).astype(np.float64) # [PYTHON]: How to convert alist into a numpy array

    return vLayerActivation
  # --------------------------------------------------------------------------------------
# ====================================================================================================  