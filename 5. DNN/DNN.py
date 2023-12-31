from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Activation, Softmax 


# =========================================================================================================================
class CDNNBasic(keras.Model):
    # --------------------------------------------------------------------------------------
    def __init__(self, p_oConfig):
        super(CDNNBasic, self).__init__(p_oConfig)
        # ..................... Object Attributes ...........................
        self.Config = p_oConfig
        
        self.ClassCount   = self.Config["DNN.LayerNeurons"][-1]
        self.LayerNeurons = self.Config["DNN.LayerNeurons"][:-1]
        self.HiddenLayers = [None]*len(self.LayerNeurons)
        self.OutputLayer  = None
        self.SoftmaxActivation = None
        self.Input        = None
        # ...................................................................
        
        if "DNN.ActivationFunction" not in self.Config:
            self.Config["DNN.ActivationFunction"] = "relu"
                    
        self.Create()
        
    # --------------------------------------------------------------------------------------
    def Create(self):
        for nIndex, nLayerNeuronCount in enumerate(self.LayerNeurons):
            self.HiddenLayers[nIndex] = Dense(nLayerNeuronCount, activation=self.Config["DNN.ActivationFunction"], use_bias=True)
        self.OutputLayer = Dense(self.ClassCount, use_bias=True)
        self.SoftmaxActivation = Softmax() 
    # --------------------------------------------------------------------------------------
    def call(self, p_tInput):
        self.Input = p_tInput
        
        # Feed forward to the next layer
        tA = p_tInput
        for oHiddenLayer in self.HiddenLayers:
            tA = oHiddenLayer(tA)

        tA = self.OutputLayer(tA)
        # Using the Softmax activation function for the neurons of the output layer 
        tA = self.SoftmaxActivation(tA)
        
        return tA    
    # --------------------------------------------------------------------------------------
# =========================================================================================================================






# =========================================================================================================================
class CDNNWithNormalization(keras.Model):
    # --------------------------------------------------------------------------------------
    def __init__(self, p_oConfig):
        super(CDNNWithNormalization, self).__init__(p_oConfig)
        # ..................... Object Attributes ...........................
        self.Config = p_oConfig
        
        self.ClassCount   = self.Config["DNN.LayerNeurons"][-1]
        self.LayerNeurons = self.Config["DNN.LayerNeurons"][:-1]
        self.HiddenLayers = [None]*len(self.LayerNeurons)
        self.NormalizationLayers = [None]*len(self.LayerNeurons)
        self.OutputLayer  = None
        self.SoftmaxActivation = None
        self.Input        = None
        # ...................................................................
        
        if "DNN.ActivationFunction" not in self.Config:
            self.Config["DNN.ActivationFunction"] = "relu"
                    
        self.Create()
        
    # --------------------------------------------------------------------------------------
    def Create(self):
        for nIndex, nLayerNeuronCount in enumerate(self.LayerNeurons):
            self.HiddenLayers[nIndex] = Dense(nLayerNeuronCount, activation=self.Config["DNN.ActivationFunction"], use_bias=True)
            self.NormalizationLayers[nIndex] = BatchNormalization()
            
        self.OutputLayer = Dense(self.ClassCount, use_bias=True)
        self.SoftmaxActivation = Softmax() 
    # --------------------------------------------------------------------------------------
    def call(self, p_tInput):
        self.Input = p_tInput
        
        # Feed forward to the next layer
        tA = p_tInput
        for nIndex, oHiddenLayer in enumerate(self.HiddenLayers):
            oNormalizationLayer = self.NormalizationLayers[nIndex]
            tA = oHiddenLayer(tA)
            tA = oNormalizationLayer(tA)

        tA = self.OutputLayer(tA)
        # Using the Softmax activation function for the neurons of the output layer 
        tA = self.SoftmaxActivation(tA)
        
        return tA    
    # --------------------------------------------------------------------------------------
# =========================================================================================================================