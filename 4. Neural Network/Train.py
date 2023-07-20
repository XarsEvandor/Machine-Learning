import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from mllib.visualization import CPlot
 
from Dataset import CRandomDataset
from Layer import CNeuralLayer


from ActivationFunctions import Relu, ReluDerivative, Sigmoid, SigmoidDerivative



# ====================================================================================================
class CMLPNeuralNetwork(list):
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self):
    # ................................................................
    self.__lastActivations = []
    self.Input       = None
    self.HiddenLayer = None
    self.OutputLayer = None
    self.Output      = None
    # ................................................................
    self.Create()
  # --------------------------------------------------------------------------------------
  def __call__(self, p_oInput):
      # Input Layer
      self.Input = p_oInput
      
      return self.Recall()
  # --------------------------------------------------------------------------------------
  def Create(self):
    # ... // Create the ML model \\ ...
    self.HiddenLayer = CNeuralLayer(HIDDEN_NEURONS, INPUT_FEATURES)
    self.OutputLayer = CNeuralLayer(1, HIDDEN_NEURONS)
    
    self.append(self.HiddenLayer)
    self.append(self.OutputLayer)
  # --------------------------------------------------------------------------------------
  def Recall(self):
    self.__lastActivations = []

    # Hidden layer
    u1 = self.HiddenLayer(self.Input)
    a1 = []
    for nIndex, u in enumerate(u1): 
        a = Sigmoid(u)
        a1.append(a)
    self.__lastActivations.append(a1)
    
    # Output layer
    u2 = self.OutputLayer(a1)
    a2 = []
    for nIndex, u in enumerate(u2): 
        a = Sigmoid(u)
        a2.append(a)
    self.__lastActivations.append(a2)

    y = a2

    return y
  # --------------------------------------------------------------------------------------
  def BackPropagateErrror(self, p_nError): # easy to read example for 2 layers
    a1, a2 = self.__lastActivations  # [PYTHON] Unpacking of multiple values from a list or a tuple

    # Calculate for the output layer
    vDelta2 = np.zeros((self.OutputLayer.NeuronCount), np.float64)
    for nThisNeuronIndex, oNeuron in enumerate(self.OutputLayer):
      vDelta2[nThisNeuronIndex] = SigmoidDerivative(a2[nThisNeuronIndex])*p_nError
      
    # Calculate for the previous layer (hidden layer)
    vDelta1 = np.zeros((self.HiddenLayer.NeuronCount), np.float64)
    for nThisNeuronIndex, oNeuron in enumerate(self.HiddenLayer):
      
      nBackPropagatedError = 0
      for nNextNeuronIndex, oNextNeuron in enumerate(oOutputLayer):
        nWeight = oNextNeuron.weights[nThisNeuronIndex]  # This is the synaptic weight between oNeuron and oNextNeuron
        nBackPropagatedError += nWeight * vDelta2[nNextNeuronIndex]
      
      #vDelta1[nThisNeuronIndex] = sigmoidDerivative(a1[nThisNeuronIndex])*nBackPropagatedError
      vDelta1[nThisNeuronIndex] = SigmoidDerivative(a1[nThisNeuronIndex])*nBackPropagatedError
      
    
    oDeltas = []    # [PYTHON] List constructor
    oDeltas.append(vDelta1)
    oDeltas.append(vDelta2)

    return oDeltas
  # --------------------------------------------------------------------------------------
  def UpdateWeights(self, p_nLearningRate, p_nDeltas): # easy to read example for 2 layers
    for nLayerIndex,oLayer in enumerate(self):
      for nNeuronIndex, oNeuron in enumerate(oLayer):
          oNeuron.TrainGradientDescent(p_nLearningRate, p_nDeltas[nLayerIndex][nNeuronIndex])
  # --------------------------------------------------------------------------------------
# ====================================================================================================


  
# _____ | Settings | ______
IS_DEBUGGING_NN_RECALL  = False
IS_PLOTING_DATA         = False

# _____ | Hyperparameters | ______
#// Architectural \\
INPUT_FEATURES              = 2
#HIDDEN_NEURONS              = 2
HIDDEN_NEURONS              = 3
#HIDDEN_NEURONS              = 5
#HIDDEN_NEURONS              = 8


# // Learning \\
MAX_EPOCH                   = 1000;
LEARNING_RATE               = 0.001;



# ... // Create the data objects \\ ...
oDataset = CRandomDataset(p_nSampleCount=200,p_nClustersPerClass=2,p_nClassSeperability=0.7)
oMinMaxScaler = preprocessing.MinMaxScaler().fit(oDataset.Samples)
oDataset.Samples = oMinMaxScaler.transform(oDataset.Samples)
print("Minmax normalized sample #1:", oDataset.Samples[0])
oDataset.Split(0.2)

# ... // Create the ML model \\ ...
oHiddenLayer = CNeuralLayer(HIDDEN_NEURONS, INPUT_FEATURES)
oOutputLayer = CNeuralLayer(1, HIDDEN_NEURONS)
oNN = CMLPNeuralNetwork()



if IS_PLOTING_DATA:
  # Plot the training set 
  oPlot = CPlot("Dataset", oDataset.Samples, oDataset.Labels)
  oPlot.Show(p_bIsMinMaxScaled=False)

  # Plot the validation set
  oPlot = CPlot("Validation Set", oDataset.VSSamples, oDataset.VSLabels)
  oPlot.Show(p_bIsMinMaxScaled=False)




oMeanError = []

# ... // Main loop for supervised training, implementing a ML algorithm \\ ...
nEpochNumber = 0;
bContinueTraining = True
while bContinueTraining:
  nEpochNumber += 1

  # Recall samples through the model to calculate error
  nPerSampleLoss = np.zeros((oDataset.TSSampleCount), dtype=np.float64)

  # We recall the whole training set
  y_true = []
  y_pred = []
  
  for nIndex in range(0, oDataset.TSSampleCount):
    # 1 sample at each step (this is Fully Stochastic Gradient Descent)
    nSample  = oDataset.TSSamples[nIndex]
    t = oDataset.TSLabels[nIndex]   # target for training
    y_true.append(t)

    # 1. RECALL
    nOutput = oNN(nSample)
    y = nOutput[0]
    if y >= 0.5:
        nPredictedClass = 1.0
    else:
        nPredictedClass = 0.0
    y_pred.append(nClass)
    
    # 2 CALCULATE ERROR 
    nSampleLoss = y - t  # The error of each sample is called loss
    nPerSampleLoss[nIndex] = nSampleLoss
    
    # 3. BACKPROPAGE ERROR AND CALCULATE GRADIENTS
    oDeltas = oNN.BackPropagateErrror(nSampleLoss)  # [PYTHON] Parentheses create a tuple that is a collection of objects
    
    # 4. UPDATE WEIGHTS USING GRADIENTS
    oNN.UpdateWeights(LEARNING_RATE, oDeltas)
    
    
  
  nTrainingSetMeanError = np.asarray(nPerSampleLoss).mean()
  oMeanError.append(nTrainingSetMeanError)

  nTrainingAccuracy = accuracy_score(y_true, y_pred)*100.0

  # Evaluating the model accuracy with the samples that are not shown during training
  y_true = []
  y_pred = []
  for nIndex in range(0, oDataset.VSSampleCount):   
    nSample  = oDataset.VSSamples[nIndex]
    t = oDataset.VSLabels[nIndex]
    y_true.append(t)
    
    nOutput = oNN(nSample)
    y = nOutput[0]
    if y >= 0.5:
        nPredictedClass = 1.0
    else:
        nPredictedClass = 0.0
    y_pred.append(nClass)
    
  nValidationAccuracy = accuracy_score(y_true, y_pred)*100.0 

  
  # Keep some stats to show
  
  print("Epoch: [%3d] | {MSE} TRN:%.6f | {ACCURACY} TRN:%.4f%% VAL:%.4f%%" % (nEpochNumber, nTrainingSetMeanError, nTrainingAccuracy, nValidationAccuracy))

  # Termination condition for training loop -> Don't stuck in an infinite training loop when there is nothing more to learn
  if (bContinueTraining):
    bContinueTraining = (nEpochNumber < MAX_EPOCH) # Simple condition when reaching a maximum of epochs

# Plot the error after the training is complete
oTrainingError = np.asarray(oMeanError, dtype=np.float64)
plt.plot(oMeanError)
plt.show()
