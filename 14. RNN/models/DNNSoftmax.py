from models.MLP import CFullyConnectedNeuralNetwork


from tensorflow import keras
from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization, Activation, Softmax
from tensorflow.keras.layers import LeakyReLU

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
    self.LayerFunctionObjects = []
  # --------------------------------------------------------------------------------------
  def DefineModel(self):                # override a virtual in our base class
    # ____// Input "Layer" \\___
    #self.Input = InputLayer(input_shape=[self.InputFeatureCount])
    #self.LayerFunctionObjects.append(self.Input)

    # // Each module of this fully Connected deep neural network has
    #   Neurons -> Activation Function -> BatchNormalization
    for nModuleNumber in range(1, self.LayerCount + 1):
      # Create the object for the synaptic integration
      oFuncObj = Dense(self.LayerNeurons[nModuleNumber], use_bias=self.HasBias, kernel_initializer="glorot_uniform", bias_initializer="zeros");
      self.LayerFunctionObjects.append(oFuncObj)
      
      if nModuleNumber != self.LayerCount:
        if self.ActivationFunction == "LeakyRelu":
          oFuncObj = LeakyReLU()
          self.LayerFunctionObjects.append(oFuncObj)
        else:
          oFuncObj = Activation(self.ActivationFunction)
          self.LayerFunctionObjects.append(oFuncObj)

        # The activation needs to be normalized if an unbounded function of the rectifier family of functions is used
        oFuncObj = BatchNormalization()
        self.LayerFunctionObjects.append(oFuncObj)
      else:
        # ____// Output layer \\___
        # For a multi-class classification it should have the softmax activation function 
        oFuncObj = Softmax()
        self.LayerFunctionObjects.append(oFuncObj)
  # --------------------------------------------------------------------------------------------------------
  def call(self, x, training=False):        # overrides a virtual in keras.Model class
    # We define a chain (pipeline) of subsequent calculations, using Keras objects that implement functions
    tA = x 
    for oFuncObject in self.LayerFunctionObjects:   
      tA = oFuncObject(tA)

    return tA
  # --------------------------------------------------------------------------------------
# =========================================================================================================================