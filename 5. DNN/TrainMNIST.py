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

# 2 Layers -> 22180 Parameters -> 0.95 Accuracy
CONFIG_BASELINE = {
            "ModelName": "MNIST1"  
           ,"DNN.InputFeatures": 28*28
           ,"DNN.LayerNeurons": [512,10]
           ,"DNN.Classes": 10
           ,"Training.MaxEpoch": 10
           ,"Training.BatchSize": 512
           ,"Training.LearningRate": 0.2
          }

CONFIG = CONFIG_BASELINE

# __________ // Create the data objects \\ __________
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

(oTSData, oVSData), oDataSetInfo = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
  
# Takes one minibatch out of the dataset. Here the size of the minibatch is the total count of samples
for tImages, tLabels in oVSData.batch(oDataSetInfo.splits['test'].num_examples).take(1):
    nImages            = tImages.numpy()
    nTargetClassLabels = tLabels.numpy()  

print("VS image features tensor shape:" , nImages.shape)
print("VS image targets vector shape :", nTargetClassLabels.shape)

if IS_PLOTING_DATA:
    for nIndex, nSample in enumerate(nImages):
      nLabel = nTargetClassLabels[nIndex]
      if (nIndex >= 0 and nIndex <= 20):
           
        if nIndex == 0:
            print("Image sample shape            :", nSample.shape)
        nImage =  nSample.astype(np.uint8) 
        plt.imshow(nImage, cmap="gray") #https://matplotlib.org/stable/tutorials/colors/colormaps.html
        #plt.imshow(nImage[4:22, 0:15, :], cmap="gray") #https://matplotlib.org/stable/tutorials/colors/colormaps.html
        plt.title("Digit %d" % nLabel)
        plt.show()    
            





    
# -----------------------------------------------------------------------------------
def NormalizeAndReshapeImage(p_tImage, p_tLabel):
    # Normalizes color component values from `uint8` to `float32`.
    tNormalizedImage = tf.cast(p_tImage, tf.float32) / 255.
    # Reshapes the 3D tensor of the image (28x28x1) into a 782-dimensional vector
    tNormalizedImage = tf.reshape(tNormalizedImage, [CONFIG["DNN.InputFeatures"]])
    # Target class labels into one-hot encoding
    tTargetOneHot = tf.one_hot(p_tLabel, CONFIG["DNN.Classes"])
    
    return tNormalizedImage, tTargetOneHot
# -----------------------------------------------------------------------------------

nBatchSize = CONFIG["Training.BatchSize"]

# Training data feed pipeline
oTSData = oTSData.map(NormalizeAndReshapeImage, num_parallel_calls=tf.data.AUTOTUNE)
oTSData = oTSData.cache()
oTSData = oTSData.shuffle(oDataSetInfo.splits['train'].num_examples)
oTSData = oTSData.batch(nBatchSize)
oTSData = oTSData.prefetch(tf.data.AUTOTUNE)
print("Training data feed object:", oTSData)

# Validation data feed pipeline
oVSData = oVSData.map(NormalizeAndReshapeImage, num_parallel_calls=tf.data.AUTOTUNE)
oVSData = oVSData.cache()
oVSData = oVSData.batch(oDataSetInfo.splits['test'].num_examples)
oVSData = oVSData.prefetch(tf.data.AUTOTUNE)
print("Validation data feed object:", oTSData)





# __________ // Create the Machine Learning model and training algorithm objects \\ __________
from DNN import CDNNBasic, CDNNWithNormalization

oNN = CDNNBasic(CONFIG)
 

nInitialLearningRate    = CONFIG["Training.LearningRate"]

oCostFunction   = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
oOptimizer      = tf.keras.optimizers.SGD(learning_rate=nInitialLearningRate)
# -----------------------------------------------------------------------------------
def LRSchedule(epoch, lr):
    if epoch == 5:
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
        
    oProcessLog = oNN.fit(  oTSData, batch_size=nBatchSize
                            ,epochs=CONFIG["Training.MaxEpoch"]
                            ,validation_data=oVSData
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
