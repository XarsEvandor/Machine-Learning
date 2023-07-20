import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from mllib.utils import RandomSeed
from clang.cindex import callbacks

# __________ | Settings | __________
IS_PLOTING_DATA         = False
IS_DEBUGABLE            = True
IS_RETRAINING           = True
RandomSeed(2022)

# __________ | Hyperparameters | __________
CONFIG_CNN = {
                 "ModelName": "CIFAR10_CNN1"
                ,"CNN.InputShape": [32,32,3]
                ,"CNN.Classes": 10
                ,"CNN.ModuleCount": 6
                ,"CNN.ConvOutputFeatures": [32,32,64,64,128,128]
                ,"CNN.ConvWindows": [ [3,2,True], [3,1,True] ,  [3,1,True], [3,2,True], [3,1,True], [3,1,True] ]
                ,"CNN.PoolWindows": [  None      , None       ,  None      , None      , [3,2]     , None      ]
                ,"CNN.HasBatchNormalization": True
                ,"Training.MaxEpoch": 24
                ,"Training.BatchSize": 128
                ,"Training.LearningRate": 0.1               
            }
                
CONFIG = CONFIG_CNN


# __________ // Create the data objects \\ __________
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from datasets.cifar10.dataset import CCIFAR10DataSet

(oTSData, oVSData), oDataSetInfo = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
  

# ... // Create the data objects \\ ...
oDataset = CCIFAR10DataSet()
print("Training samples set shape:", oDataset.TSSamples.shape)
print("Validation samples set shape:", oDataset.VSSamples.shape)

# One hot encoding for the training and validation set labels
nTSLabelsOnehot = keras.utils.to_categorical(oDataset.TSLabels)
nVSLabelsOnehot = keras.utils.to_categorical(oDataset.VSLabels)
print("Training samples one-hot-encoding targets shape:", nTSLabelsOnehot.shape)
print("Validation samples one-hot-encoding targets shape:", nVSLabelsOnehot.shape)



if IS_PLOTING_DATA:
    for nIndex, nSample in enumerate(oDataset.TSSamples):
      nLabel = oDataset.TSLabels[nIndex]
      if nIndex == 9: 
        nImage =  nSample.astype(np.uint8) # Show the kitty
        print(nImage.shape)
        print(oDataset.ClassNames[nLabel])
        plt.imshow(nImage[4:22, 0:15, :])
        plt.show()    
    
      elif nIndex == 30: # Show the toy airplane
        nImage =  nSample.astype(np.uint8)
        print(nImage.shape)
        print(oDataset.ClassNames[nLabel])
        plt.imshow(nImage)
        plt.show()      
    
    
        plt.title("Blue")
        plt.imshow(nImage[:,:,0], cmap="Blues")
        plt.show()    
        
        plt.title("Green")
        plt.imshow(nImage[:,:,1], cmap="Greens")
        plt.show()    
    
        plt.title("Red")
        plt.imshow(nImage[:,:,2], cmap="Reds")
        plt.show()                
  
            






# -----------------------------------------------------------------------------------
def NormalizeImage(p_tImage, p_tLabel):
    # Normalizes color component values from `uint8` to `float32`.
    tNormalizedImage = tf.cast(p_tImage, tf.float32) / 255.
    # Target class labels into one-hot encoding
    tTargetOneHot = tf.one_hot(p_tLabel, CONFIG["CNN.Classes"])
    
    return tNormalizedImage, tTargetOneHot
# -----------------------------------------------------------------------------------


nBatchSize = CONFIG["Training.BatchSize"]

oTSData = tf.data.Dataset.from_tensor_slices((oDataset.TSSamples, oDataset.TSLabels))
oVSData = tf.data.Dataset.from_tensor_slices((oDataset.VSSamples, oDataset.VSLabels))


# Training data feed pipeline
oTSData = oTSData.map(NormalizeImage, num_parallel_calls=tf.data.AUTOTUNE)
oTSData = oTSData.cache()
oTSData = oTSData.shuffle(oDataSetInfo.splits['train'].num_examples)
oTSData = oTSData.batch(nBatchSize)
print("Training data feed object:", oTSData)

# Validation data feed pipeline
oVSData = oVSData.map(NormalizeImage, num_parallel_calls=tf.data.AUTOTUNE)
oVSData = oVSData.batch(oDataSetInfo.splits['test'].num_examples)
print("Validation data feed object:", oVSData)





# __________ // Create the Machine Learning model and training algorithm objects \\ __________
from models.CNN import CCNNBasic

oNN = CCNNBasic(CONFIG)



# -----------------------------------------------------------------------------------
def LRSchedule(epoch, lr):
    if epoch == 10:
        nNewLR = lr * 0.5
        print("Setting LR to %.5f" % nNewLR)
        return nNewLR
    else:
        return lr
# -----------------------------------------------------------------------------------    

nInitialLearningRate    = CONFIG["Training.LearningRate"]    

oCostFunction   = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
oOptimizer = tf.keras.optimizers.SGD(learning_rate=nInitialLearningRate)
oCallbacks = [tf.keras.callbacks.LearningRateScheduler(LRSchedule)]



    
# __________ // Training Process \\ __________
sModelFolderName = CONFIG["ModelName"]
        
if (not os.path.isdir(sModelFolderName)) or IS_RETRAINING:
    oNN.compile(loss=oCostFunction, optimizer=oOptimizer, metrics=["accuracy"])

    oNN.predict(oVSData)
    
    oNN.Structure.Print("Model-Structure-%s.csv" % CONFIG["ModelName"])
    

    if IS_DEBUGABLE:
        oNN.run_eagerly = True
        
    oProcessLog = oNN.fit(  oDataset.TSSamples, nTSLabelsOnehot, batch_size=CONFIG["Training.BatchSize"]  
                            ,epochs=CONFIG["Training.MaxEpoch"]
                            ,validation_data=(oDataset.VSSamples, nVSLabelsOnehot)
                            ,callbacks=oCallbacks
                          )
    oNN.summary()          
    oNN.save(sModelFolderName)      
    
    # list all data in history
    print("Keys of Keras training process log:", oProcessLog.history.keys())
    
    
    sPrefix = sModelFolderName
            
    # summarize history for accuracy
    plt.plot(oProcessLog.history['accuracy'])
    plt.plot(oProcessLog.history['val_accuracy'])
    plt.title(sPrefix + "Accuracy")
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
    plt.title(sPrefix + sCostFunctionName + " Error")
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

      
nPredictedProbabilities = oNN.predict(oVSData)
nPredictedClassLabels  = np.argmax(nPredictedProbabilities, axis=1)


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
      




# Plot the error after the training is complete
#oTrainingError = np.asarray(oMeanError, dtype=np.float64)
#plt.plot(oMeanError)
#plt.show()
