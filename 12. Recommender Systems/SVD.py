import os
import numpy as np
import tensorflow as tf
# [PANTELIS] If we want to port an existing model from Tensorflow V1 to the Tensorflow V2 we should use this package and the tfv1 alias for any existing declaration
import tensorflow.compat.v1 as tfv1
from mllib.utils import RandomSeed
import tensorflow_ranking as tfr


from data.movielens import CMovieLens100K, CMovieLens1M, CMovieLens10M
from sklearn.model_selection import train_test_split



# __________ | Settings | __________
IS_PLOTING_DATA         = False
IS_DEBUGABLE            = True
IS_RETRAINING           = False
RandomSeed(2022)



# Loads the dataset and display some validation samples
oDataSet = CMovieLens1M()
oDataSet.Split(0.8)
print("Users: %d" % (oDataSet.UserCount))
print("Items: %d" % (oDataSet.ItemCount))
print("Posible Combinations: %d" % (oDataSet.UserCount*oDataSet.ItemCount))
print("Sample Count: %d" % (oDataSet.SampleCount))


if IS_PLOTING_DATA:
    print("-"*20, "Items", "-"*20)
    for sKey in list(oDataSet.Items.keys())[0:4]:
        print("Item ID:%s Name:%s" % (sKey, oDataSet.Items[sKey]))
        
    if oDataSet.Users is not None:    
        print("-"*20, "Users", "-"*20)    
        for sKey in list(oDataSet.Users.keys())[0:4]:    
            print("User ID:%s Genre:%s" % (sKey, oDataSet.Users[sKey]))
        
    print("-"*20, "User-Item Samples", "-"*20)
    for nIndex, oSample in enumerate(oDataSet.VSSamples[0:4]):
        sUserGenre = oDataSet.Users[ oSample[0] ]
        sItemName = oDataSet.Items[ oSample[1] ]
        nRating = oDataSet.VSLabels[nIndex]
        print("Sample:%s, Label:%0.1f" % (oSample, nRating))
        print("For User-Item combination:[%d (%s), '%s'] the ground truth rating is %.1f" % (oSample[0], sUserGenre, sItemName, nRating ))
        print("."*30)



# __________ | Hyperparameters | __________
MODEL_NUMBER = 5
CONFIG_BASELINE = {
            "ModelName": "SVD_%d" %  MODEL_NUMBER
           ,"Training.Optimizer": "MOMENTUM"
           ,"Training.MaxEpoch": 50
           ,"Training.BatchSize": 4096
           ,"Training.LearningRate": 1e-4
           ,"Training.Momentum": 0.5
           ,"Training.RegularizeL2": True
           ,"Training.WeightDecay": 3.0
          }

CONFIG = CONFIG_BASELINE


from tfcf.metrics import mae
from tfcf.metrics import rmse
from tfcf.config import Config
from tfcf.models.svd import SVD
from tfcf.models.svdpp import SVDPP

# Creates a Tensorflow v1 session. This is done behind the scenes with Keras in version above 2.0
with tfv1.Session() as sess:
    if IS_DEBUGABLE:
        tf.compat.v1.enable_eager_execution()
        
    # // Models: SVD, SVD++ \\
    # For SVD++ algorithm, if `dual` is True, then the dual term of items'. implicit feedback will be added into the original SVD++ algorithm.
    oModel = SVD(sess, p_oConfig=CONFIG, p_oDataSet=oDataSet)
    #oModel = SVDPP(sess, dual=False, p_oDataSet=oDataSet)
    #oModel = SVDPP(sess, dual=True, p_oDataSet=oDataSet)
    sModelFolder = os.path.join("MLModel", CONFIG["ModelName"])

    if not os.path.exists(sModelFolder) or IS_RETRAINING:
      nMeanRating = np.mean(oDataSet.TSLabels)
        
      oModel.train(oDataSet.TSSamples, oDataSet.TSLabels,
                   validation_data=(oDataSet.VSSamples, oDataSet.VSLabels),
                   epochs=CONFIG["Training.MaxEpoch"], batch_size=CONFIG["Training.BatchSize"], p_nMeanRating=nMeanRating)
      
      # Saves a tensorflow graph to a folder
      if not os.path.exists(sModelFolder):
        os.makedirs(sModelFolder)
      oModel.SaveModel(sModelFolder)
    else:
      # Loads a tensorflow graph from a folder
      oModel.LoadModel(sModelFolder)
    
    print("-"*30, "Predicting recommended items", "-"*30)
    y_true = oDataSet.VSLabels
    y_pred = oModel.predict(oDataSet.VSSamples)
    print("First four samples:")
    for nIndex, oSample in enumerate(oDataSet.VSSamples[0:4]):
        print("Combination User-Item:%s  Ground Truth Rating:%d    Predicted Rating:%d" % (oSample, y_true[nIndex], y_pred[nIndex]))


    print("-"*30, "Evaluating model %s" % CONFIG["ModelName"], "-"*30)
    print('rmse: {}, mae: {}'.format(rmse(y_true, y_pred), mae(y_true, y_pred)))



# Creating the user-item ratings matrix. We need the maximum ID because there might be missing IDs in the series 
oTruePerUserRankedRatings       = np.zeros( (oDataSet.MaxUserID, oDataSet.MaxItemID) , np.float32)
oPredictedPerUserRankedRatings  = np.zeros( (oDataSet.MaxUserID, oDataSet.MaxItemID) , np.float32)
for nIndex, oSample in enumerate(oDataSet.VSSamples):
    nUserID = int(oSample[0])
    nItemID = int(oSample[1])
    nRating = oDataSet.VSLabels[nIndex]
    nPredictedRating = y_pred[nIndex]
    oTruePerUserRankedRatings[nUserID, nItemID] = nRating
    oPredictedPerUserRankedRatings[nUserID, nItemID] = nPredictedRating
    
# For each user row in the array gets the sorted by rating in descending order indices, that correspond to ItemIDs.
RECOMMENDED_ITEMS = 25
FILTER_BY_CASUAL_VIEWER = True
FILTER_BY_MOVIE_CRITICS = False
nCasualViewerLimit  = 20
nMovieCriticsLimit  = 100

if FILTER_BY_CASUAL_VIEWER:
    print("--- For users that have rated a few movies (casual viewers) ---")
elif FILTER_BY_MOVIE_CRITICS:
    print("--- For users that have rated a lot of movies movies (movie critics) ---")
        
        
nSumReciprocalRank = 0.0
nCountUsers = 0
nFoundUsers = 0
for nIndex, oTrueUserRatings in enumerate(oTruePerUserRankedRatings):
    oPredictedUserRatings = oPredictedPerUserRankedRatings[nIndex,:]
    
    nCountOfRatings = np.count_nonzero(oTrueUserRatings) 
    bInclude = (nCountOfRatings > 0)
    if bInclude:
        if FILTER_BY_CASUAL_VIEWER:
            bInclude = nCountOfRatings <= nCasualViewerLimit
        elif FILTER_BY_MOVIE_CRITICS:
            bInclude = nCountOfRatings >= nMovieCriticsLimit 
    
    if bInclude: 
        oTrueTopItemsForUser = np.argsort(-oTrueUserRatings)[:RECOMMENDED_ITEMS]
        oTrueTopRatings      = oTrueUserRatings[oTrueTopItemsForUser]
        #print(oTrueTopItemsForUser)
        #print(oTrueTopRatings)

        oPredictedTopItemsForUser  = np.argsort(-oPredictedUserRatings)[:RECOMMENDED_ITEMS] 
        oPredictedTopRatings = oPredictedUserRatings[oPredictedTopItemsForUser]
        #print(oPredictedTopItemsForUser)
        #print(oPredictedTopRatings)
        
        nFound = np.where(oTrueTopItemsForUser == oPredictedTopItemsForUser[0])
        nFound = nFound[0]
        nReciprocalRank = 0.0
        if nFound:
            if len(nFound) > 0:
                nFoundUsers += 1
                nRank = float(nFound[0])+1.0
                nReciprocalRank = 1.0 / nRank
                
            if nCountUsers == 0:
                print(". "*20)
                print("Ground truth items and ratings for user %d" % (nIndex+1))
                print(oTrueTopItemsForUser)
                print(oTrueTopRatings)
                print(". "*20)
                print("Predicted items and ratings for user %d" % (nIndex+1))
                print(oPredictedTopItemsForUser)
                print(oPredictedTopRatings)
                print(". "*20)
                print("Position in ground truth of top recommendation (Rank):%.0f" % nRank)
                print(". "*20)
            
        nSumReciprocalRank += nReciprocalRank
        nCountUsers += 1
        
    
nMeanReciprocalRank = nSumReciprocalRank/ nCountUsers
print("Mean Reciprocal Rank for %d user recommendations is %.6f" % (nCountUsers, nMeanReciprocalRank ))
print("this means that the model presents the correct prediction at average rank %.0f" % (1.0 / nMeanReciprocalRank))     
print("The model recommended an item that exists in the top %d/%d items for %d users" % (RECOMMENDED_ITEMS, oDataSet.ItemCount, nFoundUsers ))
