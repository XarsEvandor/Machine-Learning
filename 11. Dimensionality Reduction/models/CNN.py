import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import InputLayer, Flatten, Dense, BatchNormalization, Activation, Softmax
from tensorflow.keras.layers import Conv2D, MaxPooling2D  


# =========================================================================================================================
class CCNNCustom(keras.Model):
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self, p_oInputShape=[32,32,3], p_nModuleCount=None, p_oConvFeatures=[], p_oWindowSizes=[], p_oPoolStrides=[], p_bHasBias=True, p_sActivationFunction="relu"):
    super(CCNNCustom, self).__init__()
    # ................................................................
    # // Attributes \\
    self.HasBias            = p_bHasBias
    self.ActivationFunction = p_sActivationFunction
    self.InputShape         = p_oInputShape
    self.InputFeatures      = np.prod(p_oInputShape)
    if p_nModuleCount is None:
      self.ModuleCount       = len(p_oConvFeatures)
    else:
      self.ModuleCount = p_nModuleCount
    self.ConvFeatures      = [self.InputFeatures] + p_oConvFeatures
    self.ConvWindowSizes   = [None] + p_oWindowSizes
    self.ConvStrides       = [None] + [1] * self.ModuleCount
    self.PoolWindowSizes   = [None] + [2] * self.ModuleCount
    self.PoolStrides       = [None] + p_oPoolStrides
    
    self.Input                = None
    self.LayerFunctionObjects = []
    # ................................................................
    self.DefineModel()
  # --------------------------------------------------------------------------------------
  def DefineModel(self):                # override a virtual in our base class
    oFuncObj = InputLayer(self.InputShape)
    self.LayerFunctionObjects.append(oFuncObj)

    for nModuleNumber in range(1, self.ModuleCount + 1):
      nFeatures   = self.ConvFeatures[nModuleNumber]

      if nModuleNumber < self.ModuleCount - 1:

        nKernelSize = self.ConvWindowSizes[nModuleNumber]
        nConvStride = self.ConvStrides[nModuleNumber]
        nPoolSize   = self.PoolWindowSizes[nModuleNumber]
        nPoolStride = self.PoolStrides[nModuleNumber]
        
        # // Basic Convolutional Module \\

        # Create the object for the synaptic integration done by a 2D convolution operation
        oFuncObj = Conv2D(nFeatures, kernel_size=nKernelSize, strides=nConvStride, use_bias=self.HasBias, padding="same", kernel_initializer="glorot_uniform", bias_initializer="zeros")
        self.LayerFunctionObjects.append(oFuncObj)
      
        # Common activation function for all neurons
        oFuncObj = Activation(self.ActivationFunction)
        self.LayerFunctionObjects.append(oFuncObj)

        # Max pooling
        oFuncObj = MaxPooling2D(pool_size=[nPoolSize, nPoolSize], strides=[nPoolStride, nPoolStride])
        self.LayerFunctionObjects.append(oFuncObj)

        # The activation needs to be normalized if an unbounded function of the rectifier family of functions is used
        oFuncObj = BatchNormalization()
        self.LayerFunctionObjects.append(oFuncObj)
      else:
        if nModuleNumber == self.ModuleCount - 1:
          oFuncObj = Flatten()
          self.LayerFunctionObjects.append(oFuncObj) 
          
        oFuncObj = Dense(nFeatures, use_bias=self.HasBias, kernel_initializer="glorot_uniform", bias_initializer="zeros");
        self.LayerFunctionObjects.append(oFuncObj)

        if nModuleNumber == self.ModuleCount - 1:
          oFuncObj = Activation(self.ActivationFunction)
          self.LayerFunctionObjects.append(oFuncObj)
        else:
          # ____// Output layer \\___
          # For a multi-class classification it should have the softmax activation function 
          oFuncObj = Softmax()
          self.LayerFunctionObjects.append(oFuncObj)
  # --------------------------------------------------------------------------------------------------------
  def call(self, inputs):        # overrides a virtual in keras.Model class
    # We define a chain (pipeline) of subsequent calculations, using Keras objects that implement functions
    tA = inputs
    for oFuncObject in self.LayerFunctionObjects:   
      tA = oFuncObject(tA)
      
    return tA
  # --------------------------------------------------------------------------------------
# =========================================================================================================================