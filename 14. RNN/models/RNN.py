import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Dense, LSTM, Softmax, Activation
from mllib.filestore import CFileStore
#from tensorflow.keras import regularizers

# =========================================================================================================================
class CRNN(keras.Model):
  # --------------------------------------------------------------------------------------
  def __init__(self, p_oConfig):
    super(CRNN, self).__init__()
    # ................................................................
    # // Attributes \\
    self.MaxInputLength     = p_oConfig["LSTM.MaxInputLength"]
    self.LSTMUnits          = p_oConfig["LSTM.Units"]
    self.RecurrentDropOut   = p_oConfig["LSTM.RecurrentDropOut"]
    self.DropOut            = p_oConfig["LSTM.DropOut"]
    self.ClassCount         = p_oConfig["ClassCount"]
    self.TimeStepInputShape = p_oConfig["LSTM.TimeStepInputShape"]
    self.TimeStepFeatures   = np.prod(self.TimeStepInputShape)

    
    self.Reshape      = None
    self.InputLayer   = None 
    self.LSTMLayer    = None
    self.OutputLayer  = None
    self.Softmax      = None
    # ................................................................
    self.CreateModel()
  # --------------------------------------------------------------------------------------
  def CreateModel(self):   
    #self.InputLayer = Dense(self.TimeStepFeatures)
    self.LSTMLayer  = LSTM(self.LSTMUnits, recurrent_dropout=self.RecurrentDropOut, dropout=self.DropOut)
    self.OutputLayer = Dense(self.ClassCount)
    self.Softmax    = Softmax()
  # --------------------------------------------------------------------------------------
  def call(self, p_tInput):
    tA = p_tInput
    #tA = self.InputLayer(tA)
    tA = self.LSTMLayer(tA)
    tA = self.OutputLayer(tA)
    tA = self.Softmax(tA) 
    
    return tA
# =========================================================================================================================                