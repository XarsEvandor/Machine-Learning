
# __________ // Create the Machine Learning model and training algorithm objects \\ __________
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from mllib.helpers import CKerasModelStructure, CModelConfig

# =========================================================================================================================
class CConvolutionalAutoencoder(keras.Model):
    # --------------------------------------------------------------------------------------
    # Constructor
    def __init__(self, p_oConfig):
        super(CConvolutionalAutoencoder, self).__init__()
        
        # ..................... Object Attributes ...........................
        self.Config = CModelConfig(self, p_oConfig)
        
        
        self.EncoderFeatures    = self.Config.Value["DA.EncoderFeatures"]
        self.DecoderFeatures    = self.Config.Value["DA.DecoderFeatures"]
        self.Downsampling       = self.Config.Value["DA.Downsampling"]
        self.Upsampling         = self.Config.Value["DA.UpSampling"]
        self.Structure = None 
        
        # ......... Keras layers .........
        self.CodeFlatteningLayer = None
        self.CodeDenseLayer      = None
        self.KerasLayers         = []
        # ...................................................................
        
        self.Config.DefaultValue("DA.ActivationFunction", "relu")
        self.Config.DefaultValue("DA.ConvHasBias", False)
        self.Config.DefaultValue("DA.HasBatchNormalization", False)
        self.Config.DefaultValue("DA.KernelInitializer", "glorot_uniform")
        self.Config.DefaultValue("DA.BiasInitializer", "zeros")
        self.Config.DefaultValue("Training.RegularizeL2", False)
        self.Config.DefaultValue("Training.WeightDecay", 1e-5)
            
        if self.Config.Value["Training.RegularizeL2"]:
            print("Using L2 regularization of weights with weight decay %.6f" % self.Config["Training.WeightDecay"])
              
        self.Create()
    # --------------------------------------------------------------------------------------------------------
    def createWeightRegulizer(self):
        if self.Config.Value["Training.RegularizeL2"]:
            oWeightRegularizer = regularizers.L2(self.Config.Value["Training.WeightDecay"])
        else:
            oWeightRegularizer = None
        return oWeightRegularizer          
    # --------------------------------------------------------------------------------------
    def Create(self):
        for nIndex,nFeatures in enumerate(self.EncoderFeatures):
            nStride = 1
            if self.Downsampling[nIndex]:
                nStride = 2
            oConvolution = layers.Conv2D(nFeatures, kernel_size=(3,3), strides=nStride, padding="same"
                                  , use_bias=self.Config.Value["DA.ConvHasBias"]
                                  , kernel_initializer=self.Config.Value["DA.KernelInitializer"]
                                  , bias_initializer=self.Config.Value["DA.BiasInitializer"]
                                  , kernel_regularizer=self.createWeightRegulizer()                            
                                  )
            self.KerasLayers.append(oConvolution)
              
            oActivation  = layers.Activation(self.Config.Value["DA.ActivationFunction"])
            self.KerasLayers.append(oActivation)
            
            if self.Config.Value["DA.HasBatchNormalization"]:
                oNormalization = layers.BatchNormalization()
                self.KerasLayers.append(oNormalization)

        #https://github.com/Seratna/TensorFlow-Convolutional-AutoEncoder
        self.CodeFlatteningLayer = layers.Flatten()
        self.KerasLayers.append(self.CodeFlatteningLayer)
        
        self.CodeDenseLayer      = layers.Dense(self.Config.Value["DA.CodeDimensions"], activation="relu")
        self.KerasLayers.append(self.CodeDenseLayer)
        
        
        nDecoderInputResolution = self.Config.Value["DA.DecoderInputResolution"]
        oDecoderFirstLayer = layers.Dense(nDecoderInputResolution[0]*nDecoderInputResolution[1]*self.DecoderFeatures[0], activation="relu")
        self.KerasLayers.append(oDecoderFirstLayer)
        
        oReshape = layers.Reshape([nDecoderInputResolution[0],nDecoderInputResolution[1],self.DecoderFeatures[0]])
        self.KerasLayers.append(oReshape)
                                
           
        for nIndex,nFeatures in enumerate(self.DecoderFeatures[1:]):
            nStride = 1
            if self.Upsampling[nIndex]:
                nStride = 2
            oDeconvolution = layers.Conv2DTranspose( nFeatures, kernel_size=(3,3), strides=nStride, padding="same"
                                              , use_bias=self.Config.Value["DA.ConvHasBias"]
                                              , kernel_initializer=self.Config.Value["DA.KernelInitializer"]
                                              , bias_initializer=self.Config.Value["DA.BiasInitializer"]
                                              , kernel_regularizer=self.createWeightRegulizer()                            
                                              )
            self.KerasLayers.append(oDeconvolution)
              
            oActivation  = layers.Activation(self.Config.Value["DA.ActivationFunction"])
            self.KerasLayers.append(oActivation)
            
            if self.Config.Value["DA.HasBatchNormalization"]:
                oNormalization = layers.BatchNormalization()
                self.KerasLayers.append(oNormalization)
                
        oLastLayerActivation = layers.Activation("sigmoid")
        self.KerasLayers.append(oLastLayerActivation)
        
    # --------------------------------------------------------------------------------------------------------
    def call(self, p_tInput):
        bPrint = self.Structure is None
        if bPrint:
            self.Structure = CKerasModelStructure()
          
        self.Input = p_tInput
        
        # ....... Convolutional Feature Extraction  .......
        # Feed forward to the next layer
        tA = p_tInput
        if bPrint:
            self.Structure.Add(tA) 
        
        for nIndex,oKerasLayer in enumerate(self.KerasLayers):
            if bPrint:
                self.Structure.Add(tA)         
            tA = oKerasLayer(tA)
        
        return tA
    # --------------------------------------------------------------------------------------------------------
# =========================================================================================================================
