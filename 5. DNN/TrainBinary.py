import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from mllib.utils import RandomSeed

# __________ | Settings | __________
IS_PLOTING_DATA         = True
IS_DEBUGABLE            = False
IS_RETRAINING           = True
RandomSeed(2022)


# __________ | Hyperparameters | __________
CONFIG_GD_LOCAL_MINIMA = {
            "ModelName": "MLP1_LOCAL_MINIMA"  
           ,"MLP.InputFeatures": 2
           ,"MLP.HiddenNeurons": 2
           ,"MLP.Classes": 2
           ,"Training.MaxEpoch": 40
           ,"Training.BatchSize": 160
           ,"Training.LearningRate": 0.1
           
          }

CONFIG_GD_GOOD = { 
            "ModelName": "MLP1_TRAIN2" 
           ,"MLP.InputFeatures": 2
           ,"MLP.HiddenNeurons": 2
           ,"MLP.Classes": 2
           ,"Training.MaxEpoch": 18
           ,"Training.BatchSize": 160
           ,"Training.LearningRate": 0.1
          }

CONFIG_FULLY_STOCHASTIC_GD_GOOD = {  
            "ModelName": "MLP1_TRAIN3"
           ,"MLP.InputFeatures": 2
           ,"MLP.HiddenNeurons": 2
           ,"MLP.Classes": 2
           ,"Training.MaxEpoch": 18
           ,"Training.BatchSize": 1
           ,"Training.LearningRate": 0.1
          }

CONFIG_STOCHASTIC_MINIBATCH_GD_GOOD = {
            "ModelName": "MLP1_TRAIN4"  
           ,"MLP.InputFeatures": 2
           ,"MLP.HiddenNeurons": 2
           ,"MLP.Classes": 2
           ,"Training.MaxEpoch": 18
           ,"Training.BatchSize": 10
           ,"Training.LearningRate": 0.15
          }


CONFIG_STOCHASTIC_MINIBATCH_GD_EXCELLENT = {
            "ModelName": "MLP1_BEST"  
           ,"MLP.InputFeatures": 2
           ,"MLP.HiddenNeurons": 2
           ,"MLP.Classes": 2
           ,"Training.MaxEpoch": 125
           ,"Training.BatchSize": 20
           ,"Training.LearningRate": 0.15
          }

CONFIG_STOCHASTIC_MINIBATCH_GD_OVERFITTING = {
           "ModelName": "MLP1_OVERFIT"  
           ,"MLP.InputFeatures": 2
           ,"MLP.HiddenNeurons": 2
           ,"MLP.Classes": 2
           ,"Training.MaxEpoch": 200
           ,"Training.BatchSize": 20
           ,"Training.LearningRate": 0.15
          }

CONFIG_STOCHASTIC_MINIBATCH_GD_UNDERFITTING = {
            "ModelName": "MLP1_UNDERFIT"  
           ,"MLP.InputFeatures": 2
           ,"MLP.HiddenNeurons": 2
           ,"MLP.Classes": 2
           ,"Training.MaxEpoch": 36
           ,"Training.BatchSize": 20
           ,"Training.LearningRate": 0.15
          }

CONFIG_MLP_4HIDDEN = {
            "ModelName": "MLP1_4HIDDEN"  
           ,"MLP.InputFeatures": 2
           ,"MLP.HiddenNeurons": 4
           ,"MLP.Classes": 2
           ,"Training.MaxEpoch": 125
           ,"Training.BatchSize": 20
           ,"Training.LearningRate": 0.15
          }

CONFIG_MLP_32HIDDEN = {
            "ModelName": "MLP1_32HIDDEN"  
           ,"MLP.InputFeatures": 2
           ,"MLP.HiddenNeurons": 32
           ,"MLP.Classes": 2
           ,"Training.MaxEpoch": 125
           ,"Training.BatchSize": 20
           ,"Training.LearningRate": 0.15
          }

CONFIG = CONFIG_STOCHASTIC_MINIBATCH_GD_EXCELLENT

# __________ // Create the data objects \\ __________
from datasets.randomdataset import CRandomDataset
from sklearn import preprocessing
from mllib.visualization import CPlot

oDataset = CRandomDataset(p_nSampleCount=200,p_nClustersPerClass=2,p_nClassSeperability=0.7)
oMinMaxScaler = preprocessing.MinMaxScaler().fit(oDataset.Samples)
oDataset.Samples = oMinMaxScaler.transform(oDataset.Samples)
print("Minmax normalized sample #1:", oDataset.Samples[0])
oDataset.Split(0.2)

if IS_PLOTING_DATA:
  # Plot the training set 
  oPlot = CPlot("Dataset", oDataset.Samples, oDataset.Labels)
  oPlot.Show(p_bIsMinMaxScaled=False)

  # Plot the validation set
  oPlot = CPlot("Validation Set", oDataset.VSSamples, oDataset.VSLabels)
  oPlot.Show(p_bIsMinMaxScaled=False)

# ... Create the Tensorflow/Keras objects for feeding the data into the training algorithm
nBatchSize = CONFIG["Training.BatchSize"]

tTSLabelsOnehot = tf.one_hot(oDataset.TSLabels, CONFIG["MLP.Classes"])
tVSLabelsOnehot = tf.one_hot(oDataset.VSLabels, CONFIG["MLP.Classes"])


print(oDataset.TSSamples.shape, oDataset.TSLabels.shape, tTSLabelsOnehot.shape)
oTrainingData = tf.data.Dataset.from_tensor_slices((oDataset.TSSamples, tTSLabelsOnehot))
#oTrainingData = oTrainingData.shuffle(oDataset.TSSampleCount).batch(nBatchSize , drop_remainder=True)

print(oDataset.VSSamples.shape, oDataset.VSLabels.shape, tVSLabelsOnehot.shape)
oValidationData = tf.data.Dataset.from_tensor_slices((oDataset.VSSamples, tVSLabelsOnehot))



# __________ // Create the Machine Learning model and training algorithm objects \\ __________
from MLP import CMLPNeuralNetwork

oNN = CMLPNeuralNetwork(CONFIG)
 

nInitialLearningRate    = CONFIG["Training.LearningRate"]

#oCostFunction   = tf.keras.losses.MeanAbsoluteError() 
#oCostFunction   = tf.keras.losses.MeanSquaredError()
oCostFunction   = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
oOptimizer      = tf.keras.optimizers.SGD(learning_rate=nInitialLearningRate)

# __________ // Training Process \\ __________
sModelFolderName = CONFIG["ModelName"]
        
if (not os.path.isdir(sModelFolderName)) or IS_RETRAINING:
    oNN.compile(loss=oCostFunction, optimizer=oOptimizer, metrics=["accuracy"])

    if IS_DEBUGABLE:
        oNN.run_eagerly = True
        
    oProcessLog = oNN.fit(  oDataset.TSSamples, tTSLabelsOnehot, batch_size=nBatchSize
                            ,epochs=CONFIG["Training.MaxEpoch"]
                            ,validation_data=(oDataset.VSSamples, tVSLabelsOnehot) 
                          )
    oNN.summary()          
    oNN.save(sModelFolderName)      
    
    # list all data in history
    print("Keys of Keras training process log:", oProcessLog.history.keys())
    
    # summarize history for accuracy
    plt.plot(oProcessLog.history['accuracy'])
    plt.plot(oProcessLog.history['val_accuracy'])
    plt.title('MLP Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # summarize history for loss
    
    sCostFunctionNameParts = oCostFunction.name.split("_")                           # [PYTHON]: Splitting string into an array of strings
    sCostFunctionNameParts = [x.capitalize() + " " for x in sCostFunctionNameParts]  # [PYTHON]: List comprehension example 
    sCostFunctionName = " ".join(sCostFunctionNameParts)                             # [PYTHON]: Joining string in a list with the space between them
    
    
    plt.plot(oProcessLog.history['loss'])
    plt.plot(oProcessLog.history['val_loss'])
    plt.title('MLP ' + sCostFunctionName + " Error")
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
else:
    # The model is trained and its state is saved (all the trainable parameters are saved). We load the model to recall the samples 
    oNN = keras.models.load_model(sModelFolderName)
    oNN.summary()    

if IS_PLOTING_DATA :
    # Plot the validation set
    oPlot = CPlot("Training Set Input Features", oDataset.TSSamples, oDataset.TSLabels)
    oPlot.Show(p_bIsMinMaxScaled=False)
    
    
    tActivation = oNN.HiddenLayer(oDataset.TSSamples)
    nTSSamplesTransformed = tActivation.numpy()
    
    
    # Plot the validation set
    oPlot = CPlot("Training Set Hidden Neuron Activations", nTSSamplesTransformed[:,0:2], oDataset.TSLabels,
                   p_sXLabel="Neuron 1", p_sYLabel="Neuron 2" )
    oPlot.Show(p_bIsMinMaxScaled=False)

    if nTSSamplesTransformed.shape[1] > 2:    
        oPlot = CPlot("Training Set Hidden Neuron Activations", nTSSamplesTransformed[:,1:3], oDataset.TSLabels,
                       p_sXLabel="Neuron 2", p_sYLabel="Neuron 3" )
        oPlot.Show(p_bIsMinMaxScaled=False)
        
        oPlot = CPlot("Training Set Hidden Neuron Activations", nTSSamplesTransformed[:,2:4], oDataset.TSLabels,
                       p_sXLabel="Neuron 3", p_sYLabel="Neuron 4" )
        oPlot.Show(p_bIsMinMaxScaled=False)

        
        
            
from mllib.evaluation import CEvaluator

      
nPredictedProbabilities = oNN.predict(oDataset.VSSamples)
nPredictedClassLabels  = np.argmax(nPredictedProbabilities, axis=1)

nTargetClassLabels     = oDataset.VSLabels      





  



# We create an evaluator object that will produce several metrics
oEvaluator = CEvaluator(nPredictedClassLabels, nTargetClassLabels)

print(oEvaluator.ConfusionMatrix)
print("Per Class Recall (Accuracy)  :", oEvaluator.Recall)
print("Per Class Precision          :", oEvaluator.Precision)
print("AverageF1Score: %.2f" % oEvaluator.AverageF1Score)      
