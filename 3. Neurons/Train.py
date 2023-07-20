import numpy as np
import matplotlib.pyplot as plt
from mllib.visualization import CPlot
from Dataset import CRandomDataset
from Neuron import CPerceptron


FEATURE_COUNT = 2       # Samples are 2D vectors, because there are 2 features

# ... // Create the data objects \\ ...
oDataset = CRandomDataset()
oDataset.Split()

# ... // Create the ML model \\ ...
oNeuron = CPerceptron(p_nDendriteCount = FEATURE_COUNT)
print("Perceptron model parameters (initial values):", oNeuron.weights, oNeuron.bias)

# Hyperparameters
MAX_EPOCH = 100;
LEARNING_RATE = 1e-3;



oMAETraining = []

# ... // Main loop for supervised training, implementing a ML algorithm \\ ...
nEpochNumber = 0;
bContinueTraining = True
while bContinueTraining:
  nEpochNumber += 1
    
  # Recall samples through the model to calculate error
  nSampleError      = np.zeros((oDataset.TSSampleCount), dtype=np.float32)
  nSumError         = 0.0
  nMeanAbsoluteError = 0.0
 
  for nIndex in range(0, oDataset.TSSampleCount):   # [C# to PYTHON]: for(int nIndex = 0; nIndex < oDataset.TSSampleCount; nIndex++)
    # 1 sample at each step
    nSample = oDataset.TSSamples[nIndex]
    t = oDataset.TSLabels[nIndex]   # target for training
    y = oNeuron.Recall(nSample);    # ground truth label for training
    nError = t-y                    # cost (error) function

    # We adjust the model for all samples according to the summed error -> This is learning!
    oNeuron.TrainPerceptron(LEARNING_RATE, nError)

    nSampleError[nIndex] = nError
    nSumError += nError
    nMeanAbsoluteError = np.sum(np.abs(nSampleError[:nIndex + 1]))/(nIndex + 1)  #  [PYTHON]: It contains a slicing operation of nSampleError vector from 0 to nIndex
    if (((nIndex % 50) == 0) or (nIndex == oDataset.TSSampleCount)): # print every 50 samples
        print(" |___        Sample: [%3d] Sample error:%.6f  MAE Error:%.6f" % (nIndex, nError, nMeanAbsoluteError))

  

  # Keep some stats to show
  oMAETraining.append(nMeanAbsoluteError)

  print("Epoch: [%3d] Mean Absolute Error:%.6f" % (nEpochNumber, nMeanAbsoluteError))

  # Termination condition for training loop -> Don't stuck in an infinite training loop when there is nothing more to learn
  if (bContinueTraining):
    bContinueTraining = (nEpochNumber < MAX_EPOCH) # Simple condition when reaching a maximum of epochs
    
    
# Plot the error after the training is complete
oTrainingError = np.asarray(oMAETraining, dtype=np.float32)
plt.plot(oMAETraining)
plt.show()


print("Perceptron model parameters (learned values):", oNeuron.weights, oNeuron.bias)

nSlope = -oNeuron.weights[0] / oNeuron.weights[1]
nIntercept = -oNeuron.bias / oNeuron.weights[1]


# Plot the decision line on the whole dataset
oPlot = CPlot("Dataset", oDataset.Samples, oDataset.Labels)
oPlot.Show(p_bIsMinMaxScaled=False, p_nLineSlope=nSlope, p_nLineIntercept=nIntercept)

# Plot the decision line on the validation set
oPlot = CPlot("Validation Set", oDataset.VSSamples, oDataset.VSLabels)
oPlot.Show(p_bIsMinMaxScaled=False, p_nLineSlope=nSlope, p_nLineIntercept=nIntercept)    