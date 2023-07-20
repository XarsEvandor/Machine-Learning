from models.MLP import CFullyConnectedNeuralNetwork


from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Activation, Softmax


# =========================================================================================================================
class CFullyConnectedDNN(CFullyConnectedNeuralNetwork):
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self, p_nInputFeatureCount=2, p_oLayerNeurons=[4,1], p_bHasBias=True, p_sActivationFunction="sigmoid"):
    super(CFullyConnectedDNN, self).__init__(p_nInputFeatureCount, p_oLayerNeurons)   # [PYTHON]his is the call to the ancestor's constructor  ( [C#] : base(p_nFeatureCount, p_oLayerNeurons) )
    # ................................................................
    # // Attributes \\
    self.HasBias            = p_bHasBias
    self.ActivationFunction = p_sActivationFunction;
    # ................................................................
    self.LayerFunctionObjects = [None]*(self.LayerCount*2) # [PYTHON]: Initialize an array with self.LayerCount x 2 items that all are set to None ( [C#] null)
  # --------------------------------------------------------------------------------------
  def DefineModel(self):                # override a virtual in our base class
    # ................ Input "Layer" ................
    self.Input = Input(shape=[self.InputFeatureCount])
    self.LayerFunctionObjects[0] = self.Input

    for nLayerNumber in range(1, self.LayerCount + 1):
      nBaseIndex = (nLayerNumber - 1)*2
      # Create the object for the synaptic integration
      self.LayerFunctionObjects[nBaseIndex] = Dense(self.LayerNeurons[nLayerNumber], use_bias=self.HasBias, kernel_initializer="glorot_uniform", bias_initializer="zeros");
      # Create the object for the activation function
      self.LayerFunctionObjects[nBaseIndex + 1] = Activation(self.ActivationFunction)

  # --------------------------------------------------------------------------------------------------------
  def call(self, x, training=False):        # overrides a virtual in keras.Model class

    #oFunctionObjects = self.LayerFunctionObjects[1:] # [PYTHON]: Skip the item at the first index in the list and enumerate on the rest of them

    # We define a chain (pipeline) of subsequent calculations
    tA = x
    for oFunction in self.LayerFunctionObjects:   
      tA = oFunction(tA)

    return tA
  # --------------------------------------------------------------------------------------
# =========================================================================================================================