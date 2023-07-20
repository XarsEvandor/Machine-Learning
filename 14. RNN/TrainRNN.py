import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


from mllib.utils import RandomSeed
from models.RNN import CRNN
# What tensorflow version is installed in this VM?
print("Tensorflow version " + tf.__version__)
print("|__ GPUs Available: ", tf.config.list_physical_devices('GPU'))
print("|__ Default GPU Device:%s" % tf.test.gpu_device_name())

# __________ | Settings | __________
IS_PLOTING_DATA         = False
IS_DEBUGABLE            = True
IS_RETRAINING           = True
RandomSeed(2022)



# __________ | Hyperparameters | __________
CONFIG_RNN = {
                 "ModelName": "RNN_SIGNAL"
                ,"ClassCount"                   : 20
                ,"LSTM.TimeStepInputShape"      : [49, 7]
                ,"LSTM.MaxInputLength"          : 50
                ,"LSTM.Units"                   : 50
                ,"LSTM.RecurrentDropOut"        : 0.2
                ,"LSTM.DropOut"                 : 0.2
                             
                ,"Training.MaxEpoch"    : 200
                ,"Training.BatchSize"   : 20
                #,"Training.LearningRate": 0.01
                ,"Training.LearningRate": 0.5
                ,"Training.LearningRateScheduling": [[50, 0.2], [100, 0.1]]
                ,"Training.Momentum": 0.9             
            }
                
CONFIG = CONFIG_RNN
                
                
# __________ // Create the data objects \\ __________
from data.NeuroStatus import CNeuroStatus



# ... // Create the data objects \\ ...
oDataset = CNeuroStatus()
oDataset.SplitSequences(25)
print("-"*40, oDataset.Name, "-"*40)
  
print("%d sequences with %d clips, each one %d data points with %d features" % oDataset.Samples.shape)
print("%d labels for sequences" % oDataset.Labels.shape)

print("."*20, "Training Set", "."*40)
print("%d sequences with %d clips, each one %d data points with %d features" % oDataset.TSSamples.shape)
print("%d labels for sequences" % oDataset.TSLabels.shape)

print("."*20, "Validation Set", "."*40)
print("%d sequences with %d clips, each one %d data points with %d features" % oDataset.VSSamples.shape)
print("%d labels for sequences" % oDataset.VSLabels.shape)

nTSLabelsOnehot = keras.utils.to_categorical(oDataset.TSLabels, num_classes=CONFIG["ClassCount"])
nVSLabelsOnehot = keras.utils.to_categorical(oDataset.VSLabels, num_classes=CONFIG["ClassCount"])

nBatchSize = CONFIG["Training.BatchSize"] 

nNewShape = list(oDataset.TSSamples.shape)
nFeatures = np.prod(nNewShape[-2:])
nNewShape = nNewShape[:-2] + [nFeatures]

oDataset.TSSamples = oDataset.TSSamples.reshape(nNewShape)
oDataset.VSSamples = oDataset.VSSamples.reshape(nNewShape)

print(oDataset.TSSamples.shape, oDataset.VSSamples.shape)

oTSData = tf.data.Dataset.from_tensor_slices((oDataset.TSSamples, nTSLabelsOnehot))
oTSData = oTSData.shuffle(nBatchSize).batch(nBatchSize, drop_remainder=True)

oVSData = tf.data.Dataset.from_tensor_slices((oDataset.VSSamples, nVSLabelsOnehot))
oVSData = oVSData.batch(oDataset.VSSampleCount, drop_remainder=True)


oNN = CRNN(CONFIG)

# -----------------------------------------------------------------------------------
def LRSchedule(epoch, lr):
    nNewLR = lr
    for nIndex,oSchedule in enumerate(CONFIG["Training.LearningRateScheduling"]):
        if epoch == oSchedule[0]:
            nNewLR = oSchedule[1]
            print("Schedule #%d: Setting LR to %.5f" % (nIndex+1,nNewLR))
            break
    return nNewLR
# -----------------------------------------------------------------------------------   

nInitialLearningRate    = CONFIG["Training.LearningRate"]  

oCostFunction   = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
#oCostFunction   = tf.keras.losses.MeanSquaredError()
#oCostFunction   = keras.losses.BinaryCrossentropy()
#oMetrics = [keras.metrics.CategoricalAccuracy(name="average_accuracy", dtype=None)]
oMetrics = ["accuracy"] 
#oMetrics = [keras.metrics.BinaryAccuracy(name="average_accuracy", threshold=0.5, dtype=None)]


if "Training.LearningRateScheduling" in CONFIG:
  oOptimizer = tf.keras.optimizers.SGD(learning_rate=nInitialLearningRate, momentum=CONFIG["Training.Momentum"])
  oCallbacks = [tf.keras.callbacks.LearningRateScheduler(LRSchedule)]
else:
  oOptimizer = keras.optimizers.RMSprop(learning_rate=nInitialLearningRate)
  oCallbacks = None


    
# __________ // Training Process \\ __________
sModelFolderName = CONFIG["ModelName"]
        
if (not os.path.isdir(sModelFolderName)) or IS_RETRAINING:
    oNN.compile(loss=oCostFunction, optimizer=oOptimizer, metrics=oMetrics)
    if IS_DEBUGABLE:
        oNN.run_eagerly = True
        
    oNN.predict(oVSData)
    #oNN.Structure.Print("Model-Structure-%s.csv" % CONFIG["ModelName"])
        
    oProcessLog = oNN.fit(  oTSData, batch_size=nBatchSize
                            ,epochs=CONFIG["Training.MaxEpoch"]
                            ,validation_data=oVSData
                            ,callbacks=oCallbacks 
                          )
    oNN.summary()          
    oNN.save(sModelFolderName)      
else:
    # The model is trained and its state is saved (all the trainable parameters are saved). We load the model to recall the samples 
    oNN = keras.models.load_model(sModelFolderName)
    oNN.summary()    
    oProcessLog = None

        
        
if oProcessLog is not None: # [PYTHON] Checks that object reference is not Null
    sModelPrefix = "RNN"
    # list all data in history
    print("Keys of Keras training process log:", oProcessLog.history.keys())
    
    # Plot the accuracy during the training epochs
    plt.plot(oProcessLog.history['accuracy'])
    plt.plot(oProcessLog.history['val_accuracy'])
    plt.title('%s Accuracy' % sModelPrefix)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # Plot the error during the training epochs
    sCostFunctionNameParts = oCostFunction.name.split("_")                           # [PYTHON]: Splitting string into an array of strings
    sCostFunctionNameParts = [x.capitalize() + " " for x in sCostFunctionNameParts]  # [PYTHON]: List comprehension example 
    sCostFunctionName = " ".join(sCostFunctionNameParts)                             # [PYTHON]: Joining string in a list with the space between them
    
    plt.plot(oProcessLog.history['loss'])
    plt.plot(oProcessLog.history['val_loss'])
    plt.title('%s %s' % (sModelPrefix, sCostFunctionName) + " Error")
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
from mllib.evaluation import CEvaluator
from mllib.visualization import CPlotConfusionMatrix            
            
     
nPredictedProbabilities = oNN.predict(oVSData)
nPredictedClassLabels  = np.argmax(nPredictedProbabilities, axis=1)



oEvaluator = CEvaluator(oDataset.VSLabels, nPredictedClassLabels)

oEvaluator.PrintConfusionMatrix()
print("Per Class Recall (Accuracy)  :", oEvaluator.Recall)
print("Per Class Precision          :", oEvaluator.Precision)
print("Average Accuracy: %.4f" % oEvaluator.AverageRecall)
print("Average F1 Score: %.4f" % oEvaluator.AverageF1Score)
      
oConfusionMatrixPlot = CPlotConfusionMatrix(oEvaluator.ConfusionMatrix)
oConfusionMatrixPlot.Show() 


