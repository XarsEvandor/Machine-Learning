from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Activation, Softmax 


# =========================================================================================================================
class CMLPNeuralNetwork(keras.Model):
    # --------------------------------------------------------------------------------------
    def __init__(self, p_oConfig):
        super(CMLPNeuralNetwork, self).__init__(p_oConfig)
        # ..................... Object Attributes ...........................
        self.Config = p_oConfig
        
        self.HiddenLayer = None
        self.OutputLayer = None
        
        self.Input       = None
        # ...................................................................
        
        if "MLP.ActivationFunction" not in self.Config:
            self.Config["MLP.ActivationFunction"] = "sigmoid"
                    
        self.Create()
        
    # --------------------------------------------------------------------------------------
    def Create(self):
        self.HiddenLayer = Dense(self.Config["MLP.HiddenNeurons"], activation=self.Config["MLP.ActivationFunction"], use_bias=True)
        self.OutputLayer = Dense(self.Config["MLP.Classes"]      , activation=self.Config["MLP.ActivationFunction"], use_bias=True)
    # --------------------------------------------------------------------------------------
    def call(self, p_tInput):
        self.Input = p_tInput
        
        tA = self.HiddenLayer(p_tInput)
        tA = self.OutputLayer(tA)
        
        return tA    
    # --------------------------------------------------------------------------------------
# =========================================================================================================================