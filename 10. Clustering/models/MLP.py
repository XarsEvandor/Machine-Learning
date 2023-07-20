from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Activation, Softmax


# ====================================================================================================
class CFullyConnectedNeuralNetwork(keras.Model):
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self, p_nInputFeatureCount, p_oLayerNeurons):
    super(CFullyConnectedNeuralNetwork, self).__init__()            # [PYTHON]his is the call to the ancestor's constructor  ( [C#] : base() )
    # ................................................................
    # // Attributes \\
    self.InputFeatureCount = p_nInputFeatureCount
    self.LayerNeurons      = [self.InputFeatureCount] + p_oLayerNeurons  # [PYTHON] Union of two sets. Here we want indexing of layer neurons to start from 1
    self.LayerCount        = len(p_oLayerNeurons)
    
    self.Input                = None
    self.LayerFunctionObjects = []
    # ................................................................
  # --------------------------------------------------------------------------------------------------------  
  def DefineModel(self): # [PYTHON] In python all methods are "virtual". Dynamic binding is enabled for any method
    pass                 # [PYTHON] This is needed to initialize an empty method
  # --------------------------------------------------------------------------------------------------------  

# ====================================================================================================    







# ====================================================================================================
class CMLPNeuralNetwork(CFullyConnectedNeuralNetwork):
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self, p_nInputFeatureCount=2, p_oLayerNeurons=[4,1], p_bHasBias=True):
    super(CMLPNeuralNetwork, self).__init__(p_nInputFeatureCount, p_oLayerNeurons)   # [PYTHON]his is the call to the ancestor's constructor  ( [C#] : base(p_nFeatureCount, p_oLayerNeurons) )
    # ................................................................
    # // Attributes \\
    self.HasBias = p_bHasBias
    # ................................................................
    
    self.LayerFunctionObjects = [None]*(self.LayerCount*2) # [PYTHON]: Initialize an array with self.LayerCount x 2 items that all are set to None ( [C#] null) 
  # --------------------------------------------------------------------------------------------------------  
  def DefineModel(self):                # override a virtual in our base class
    # ................ Input "Layer" ................
    self.Input = Input(shape=[self.InputFeatureCount])
    

    # ................ Hidden Layer ................
    nLayerNumber = 1
    nBaseIndex = (nLayerNumber - 1)*2
    # Create the object for the synaptic integration
    self.LayerFunctionObjects[nBaseIndex] = Dense(self.LayerNeurons[nLayerNumber], use_bias=self.HasBias, kernel_initializer="glorot_uniform", bias_initializer="zeros");
    # Create the object for the activation function
    self.LayerFunctionObjects[nBaseIndex + 1] = Activation("sigmoid")

    # ................ Output Layer ................
    nLayerNumber = 2
    nBaseIndex = (nLayerNumber - 1)*2
    # Create the object for the synaptic integration
    self.LayerFunctionObjects[nBaseIndex] = Dense(self.LayerNeurons[nLayerNumber], use_bias=self.HasBias, kernel_initializer="glorot_uniform", bias_initializer="zeros");
    # Create the object for the activation function
    self.LayerFunctionObjects[nBaseIndex + 1] = Activation("sigmoid")

  # --------------------------------------------------------------------------------------------------------
  def call(self, x, training=False):        # overrides a virtual in keras.Model class

    # We define a chain (pipeline) of activations, starting from the input ->  synaptic integration -> activation functions -> next layer ...
    tA = x
    tA = self.LayerFunctionObjects[0](tA)  # tA = w1_t * tA + b1
    tA = self.LayerFunctionObjects[1](tA)  # tA = sigmoid(tA)
    tA = self.LayerFunctionObjects[2](tA)  # tA = w2_t * tA + b2
    tA = self.LayerFunctionObjects[3](tA)  # tA = sigmoid(tA)

    return tA

# ====================================================================================================