import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from mllib.utils import RandomSeed
from clang.cindex import callbacks

# __________ | Settings | __________
IS_PLOTING_DATA         = False
IS_DEBUGABLE            = False
IS_RETRAINING           = False
RandomSeed(2022)


# __________ | Hyperparameters | __________

# 2 Layers -> 22180 Parameters -> 0.95 Accuracy
CONFIG_BASELINE = {
            "ModelName": "QPEDS2"  
           ,"DNN.InputFeatures": 72
           ,"DNN.LayerNeurons": [288, 4]
           ,"DNN.Classes": 4
           ,"Training.MaxEpoch": 200
           ,"Training.BatchSize": 160
           ,"Training.LearningRate": 0.2
          }

# 3 Layers -> 10804 Parameters -> 0.97 Accuracy
CONFIG_GOOD_3 = {
            "ModelName": "QPEDS3"  
           ,"DNN.InputFeatures": 72
           ,"DNN.LayerNeurons": [72,72,4]
           ,"DNN.Classes": 4
           ,"Training.MaxEpoch": 400
           ,"Training.BatchSize": 160
           ,"Training.LearningRate": 0.1
          }



# 4 Layers -> 1780 Parameters -> 0.98 Accuracy
CONFIG_BEST_4 = {
            "ModelName": "QPEDS4"  
           ,"DNN.InputFeatures": 72
           ,"DNN.LayerNeurons": [16,16,16,4]
           ,"DNN.Classes": 4
           ,"Training.MaxEpoch": 400
           ,"Training.BatchSize": 160
           ,"Training.LearningRate": 0.1
          }
'''
[[47  1  0  0]
 [ 2 52  0  0]
 [ 0  0 39  1]
 [ 0  0  1 57]]
Per Class Recall (Accuracy)  : [0.97916667 0.96296296 0.975      0.98275862]
Per Class Precision          : [0.95918367 0.98113208 0.975      0.98275862]
AverageF1Score: 0.98

'''

# 5 Layers -> 2052 Parameters -> 0.98 Accuracy
CONFIG_BEST_5 = {
            "ModelName": "QPEDS5"  
           ,"DNN.InputFeatures": 72
           ,"DNN.LayerNeurons": [16,16,16,16,4]
           ,"DNN.Classes": 4
           ,"Training.MaxEpoch": 400
           ,"Training.BatchSize": 120
           ,"Training.LearningRate": 0.1
          }

'''
[[46  1  0  0]
 [ 3 52  0  0]
 [ 0  0 40  1]
 [ 0  0  0 57]]
Per Class Recall (Accuracy)  : [0.9787234  0.94545455 0.97560976 1.        ]
Per Class Precision          : [0.93877551 0.98113208 1.         0.98275862]
AverageF1Score: 0.98
'''
CONFIG = CONFIG_BEST_5

# __________ // Create the data objects \\ __________

from datasets.quadrapeds import CQuadrapedsDataSet
from sklearn import preprocessing
from mllib.visualization import CPlot

oDataset = CQuadrapedsDataSet(1000)
oMinMaxScaler = preprocessing.MinMaxScaler().fit(oDataset.Samples)
oDataset.Samples = oMinMaxScaler.transform(oDataset.Samples)
print("Minmax normalized sample #1:", oDataset.Samples[0])
oDataset.Split(0.2)

if IS_PLOTING_DATA:
  # Plot the training set 
  oPlot = CPlot("Dataset", oDataset.Samples[:,6:8], oDataset.Labels
                ,p_sLabelDescriptions=oDataset.ClassNames, p_sColors=sColorScheme
                ,p_sXLabel="Feature 6", p_sYLabel="Feature 7"
                )
  oPlot.Show(p_bIsMinMaxScaled=False)
                 
                 
  # Plot the validation set
  oPlot = CPlot("Validation Set", oDataset.VSSamples[:,6:8], oDataset.VSLabels
                ,p_sLabelDescriptions=oDataset.ClassNames, p_sColors=sColorScheme
                ,p_sXLabel="Feature 6", p_sYLabel="Feature 7"
                )
  oPlot.Show(p_bIsMinMaxScaled=False)

# ... Create the Tensorflow/Keras objects for feeding the data into the training algorithm
nBatchSize = CONFIG["Training.BatchSize"]

tTSLabelsOnehot = tf.one_hot(oDataset.TSLabels, CONFIG["DNN.Classes"])
tVSLabelsOnehot = tf.one_hot(oDataset.VSLabels, CONFIG["DNN.Classes"])

print(oDataset.TSSamples.shape, oDataset.TSLabels.shape, tTSLabelsOnehot.shape)
oTrainingData = tf.data.Dataset.from_tensor_slices((oDataset.TSSamples, tTSLabelsOnehot))
oTrainingData = oTrainingData.shuffle(oDataset.TSSampleCount).batch(nBatchSize, drop_remainder=True)

print(oDataset.VSSamples.shape, oDataset.VSLabels.shape, tVSLabelsOnehot.shape)
oValidationData = tf.data.Dataset.from_tensor_slices((oDataset.VSSamples, tVSLabelsOnehot))




# __________ // Create the Machine Learning model and training algorithm objects \\ __________
from DNN import CFullyConnectedDNN

oNN = CDNNBasic(CONFIG)
 

nInitialLearningRate    = CONFIG["Training.LearningRate"]

oCostFunction   = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
oOptimizer      = tf.keras.optimizers.SGD(learning_rate=nInitialLearningRate)

# -----------------------------------------------------------------------------------
def LRSchedule(epoch, lr):
    if epoch == 100:
        nNewLR = lr * 0.5
        print("Setting LR to %.5f" % nNewLR)
        return nNewLR
    elif epoch == 200:
        nNewLR = lr * 0.5
        print("Setting LR to %.5f" % nNewLR)
        return nNewLR
    elif epoch == 300:
        nNewLR = lr * 0.5
        print("Setting LR to %.5f" % nNewLR)
        return nNewLR
    else:
        return lr
# -----------------------------------------------------------------------------------    
    
oLearningRateSchedule = tf.keras.callbacks.LearningRateScheduler(LRSchedule)
    
# __________ // Training Process \\ __________
sModelFolderName = CONFIG["ModelName"]
        
if (not os.path.isdir(sModelFolderName)) or IS_RETRAINING:
    oNN.compile(loss=oCostFunction, optimizer=oOptimizer, metrics=["accuracy"])

    if IS_DEBUGABLE:
        oNN.run_eagerly = True
        
    oProcessLog = oNN.fit(  oTrainingData, batch_size=nBatchSize
                            ,epochs=CONFIG["Training.MaxEpoch"]
                            ,validation_data=(oDataset.VSSamples, tVSLabelsOnehot)
                            ,callbacks=[oLearningRateSchedule] 
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
    oPlot = CPlot("Training Set Input Features", oDataset.TSSamples[:,6:8], oDataset.TSLabels
                  ,p_sLabelDescriptions=oDataset.ClassNames, p_sColors=sColorScheme 
                  ,p_sXLabel="Feature 6", p_sYLabel="Feature 7" 
                  )
    oPlot.Show(p_bIsMinMaxScaled=False)
    
    
    tActivation = oNN.HiddenLayer(oDataset.TSSamples)
    nTSSamplesTransformed = tActivation.numpy()
    
    # Plot the validation set
    oPlot = CPlot("Training Set Hidden Neuron Activations", nTSSamplesTransformed[:,:2], oDataset.TSLabels
                  ,p_sLabelDescriptions=oDataset.ClassNames, p_sColors=sColorScheme                  
                  ,p_sXLabel="Neuron 1", p_sYLabel="Neuron 2" )
    oPlot.Show(p_bIsMinMaxScaled=False)

    if nTSSamplesTransformed.shape[1] > 2:    
        oPlot = CPlot("Training Set Hidden Neuron Activations", nTSSamplesTransformed[:,1:3], oDataset.TSLabels
                      ,p_sLabelDescriptions=oDataset.ClassNames, p_sColors=sColorScheme                      
                      ,p_sXLabel="Neuron 2", p_sYLabel="Neuron 3" )
        oPlot.Show(p_bIsMinMaxScaled=False)
        
        oPlot = CPlot("Training Set Hidden Neuron Activations", nTSSamplesTransformed[:,2:4], oDataset.TSLabels
                      ,p_sLabelDescriptions=oDataset.ClassNames, p_sColors=sColorScheme
                      ,p_sXLabel="Neuron 3", p_sYLabel="Neuron 4" )
        oPlot.Show(p_bIsMinMaxScaled=False)

        
        
            
from mllib.evaluation import CEvaluator

      
nPredictedProbabilities = oNN.predict(oDataset.VSSamples)
nPredictedClassLabels  = np.argmax(nPredictedProbabilities, axis=1)

nTargetClassLabels     = oDataset.VSLabels      





  

from visualization import CPlotConfusionMatrix

# We create an evaluator object that will produce several metrics
oEvaluator = CEvaluator(nTargetClassLabels, nPredictedClassLabels)

oEvaluator.PrintConfusionMatrix()

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(oEvaluator.ConfusionMatrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(oEvaluator.ConfusionMatrix.shape[0]):
    for j in range(oEvaluator.ConfusionMatrix.shape[1]):
        ax.text(x=j, y=i,s=oEvaluator.ConfusionMatrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predicted Label', fontsize=18)
plt.ylabel('Actual Label', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()



print("Per Class Recall (Accuracy)  :", oEvaluator.Recall)
print("Per Class Precision          :", oEvaluator.Precision)
print("Average Accuracy: %.4f" % oEvaluator.AverageRecall)
print("Average F1 Score: %.4f" % oEvaluator.AverageF1Score)
      
oConfusionMatrixPlot = CPlotConfusionMatrix(oEvaluator.ConfusionMatrix)
oConfusionMatrixPlot.Show()      




# Plot the error after the training is complete
#oTrainingError = np.asarray(oMeanError, dtype=np.float64)
#plt.plot(oMeanError)
#plt.show()
