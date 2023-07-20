import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from mllib.utils import RandomSeed
from clang.cindex import callbacks
from mllib.visualization import CPlot
# __________ | Settings | __________
IS_PLOTING_DATA         = False
IS_DEBUGABLE            = True
IS_RETRAINING           = True
RandomSeed(2022)

# __________ | Hyperparameters | __________
CONFIG_DA1 = {
                 "ModelName": "MNIST_DA1"
                ,"DA.InputShape": [28,28,1]
                ,"DA.EncoderFeatures": [64,64]
                ,"DA.CodeDimensions" : 32
                ,"DA.Downsampling"   : [True,True]
                ,"DA.DecoderFeatures": [64,64,1]
                ,"DA.DecoderInputResolution": [7,7]
                ,"DA.UpSampling"     :  [True,True]
                ,"DA.HasBatchNormalization": True
                #,"Training.RegularizeL2": True
                #,"Training.WeightDecay": 1e-3                 
                ,"Training.MaxEpoch": 20
                ,"Training.BatchSize": 500
                ,"Training.LearningRate": 1e-3
                #,"Training.LearningRateScheduling": [[5, 1e-4], [10, 5e-5], [15, 2e-5]]
                #,"Training.Momentum": 0.9    
            }

CONFIG_DA2 = {
                 "ModelName": "MNIST_DA2"
                ,"DA.InputShape": [28,28,1]
                ,"DA.EncoderFeatures": [32,32,32,32]
                ,"DA.CodeDimensions" : 20
                ,"DA.Downsampling"   : [True,False,True,False]
                ,"DA.DecoderFeatures": [32,32,32,32,1]
                ,"DA.DecoderInputResolution": [7,7]
                ,"DA.UpSampling"     :  [False,True,False,True]
                ,"DA.HasBatchNormalization": True
                #,"Training.RegularizeL2": True
                #,"Training.WeightDecay": 1e-3                 
                ,"Training.MaxEpoch": 20
                ,"Training.BatchSize": 500
                ,"Training.LearningRate": 1e-3
            }
                
CONFIG = CONFIG_DA1


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
def NormalizeImage(p_tImage, p_tLabel):
    # Normalizes color component values from `uint8` to `float32`.
    tNormalizedImage = tf.cast(p_tImage, tf.float32) / 255.
    # Target class labels into one-hot encoding
    tTargetImage = tNormalizedImage 
    
    
    return tNormalizedImage, tTargetImage
# -----------------------------------------------------------------------------------


nBatchSize = CONFIG["Training.BatchSize"]

# Training data feed pipeline

oTSData = oTSData.map(NormalizeImage, num_parallel_calls=tf.data.AUTOTUNE)
oTSData = oTSData.cache()
oTSData = oTSData.shuffle(oDataSetInfo.splits['train'].num_examples)
oTSData = oTSData.batch(nBatchSize)
oTSData = oTSData.prefetch(tf.data.AUTOTUNE)
print("Training data feed object:", oTSData)

# Validation data feed pipeline
oVSData = oVSData.map(NormalizeImage, num_parallel_calls=tf.data.AUTOTUNE)
#oVSData = oVSData.cache()
oVSData = oVSData.batch(oDataSetInfo.splits['test'].num_examples)
#oVSData = oVSData.prefetch(tf.data.AUTOTUNE)
print("Validation data feed object:", oVSData)





# __________ // Create the Machine Learning model and training algorithm objects \\ __________
from models.DA import CConvolutionalAutoencoder


oNN = CConvolutionalAutoencoder(CONFIG)

    
 





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
  

oCostFunction   = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
oOptimizer = tf.keras.optimizers.Adam(nInitialLearningRate)
oCallbacks = None
#oOptimizer = tf.keras.optimizers.SGD(learning_rate=nInitialLearningRate, momentum=CONFIG["Training.Momentum"])
#oCallbacks = [tf.keras.callbacks.LearningRateScheduler(LRSchedule)]



# __________ // Training Process \\ __________
sModelFolderName = CONFIG["ModelName"]
        
if (not os.path.isdir(sModelFolderName)) or IS_RETRAINING:
    oNN.compile( loss=oCostFunction, optimizer=oOptimizer
                , metrics=[tf.keras.metrics.RootMeanSquaredError()])

    oNN.predict(oVSData)
    oNN.Structure.Print("Model-Structure-%s.csv" % CONFIG["ModelName"])
    
    if IS_DEBUGABLE:
        oNN.run_eagerly = True
        
    oProcessLog = oNN.fit(  oTSData, batch_size=nBatchSize
                            ,epochs=CONFIG["Training.MaxEpoch"]
                            ,validation_data=oVSData
                            ,callbacks=oCallbacks 
                          )
    oNN.summary()          
    oNN.save(sModelFolderName)      
    
    # list all data in history
    print("Keys of Keras training process log:", oProcessLog.history.keys())
    
    sPrefix = "DA "
            
    # summarize history for accuracy
    sMetricName = "root_mean_squared_error"
    plt.plot(oProcessLog.history[sMetricName])
    plt.plot(oProcessLog.history["val_" + sMetricName])
    plt.title(sPrefix + sMetricName)
    plt.ylabel(sMetricName)
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

        
        
# Takes one minibatch out of the dataset. Here the size of the minibatch is the total count of samples
for tImages, tLabels in oVSData.take(1):
    nImages            = tImages.numpy()
    nTargetClassLabels = tLabels.numpy()  

print("VS image features tensor shape:" , nImages.shape)
print("VS image targets vector shape :" , nTargetClassLabels.shape)



nOutputImages = oNN.predict(nImages)



for nIndex, nOutputImage in enumerate(nOutputImages[0:10,:]):
    nOriginalImage = nImages[nIndex]*255.0
    nReconstructedImage = nOutputImage*255.0
    
    plt.imshow(nOriginalImage.astype(np.uint8), cmap="gray") 
    plt.title("Original image samples #%d" % (nIndex+1))
    plt.show()  
    
             
    plt.imshow(nReconstructedImage.astype(np.uint8), cmap="gray") 
    plt.title("Reconstructed image samples #%d" % (nIndex+1))
    plt.show()    

