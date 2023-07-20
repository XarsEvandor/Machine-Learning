import os
  
import numpy as np
import pandas as pd
import requests

from io import BytesIO
from io import StringIO
from zipfile import ZipFile


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    sys.path.append("..")
    os.chdir("..")

    

from mllib.data import CCustomDataSet
from mllib.filestore import CFileStore




# =========================================================================================================================
class CFoodcomRatingsDataset(CCustomDataSet):
    # ------------------------------------------------------------------------------------
    def __init__(self, p_sCode=None, p_bIsVerbose=False):
        super(CFoodcomRatingsDataset, self).__init__()
        # ................................................................
        # // Fields \\
        self.Code            = p_sCode
        self.IsVerbose       = p_bIsVerbose
        self.RatingsFile     = None
        self.RatingsFileRaw  = None
        self.ItemsFile       = None
        self.ItemsFileRaw    = None
        self.UsersFile       = None
        self.UsersFileRaw    = None
        self.DownloadURL     = None
        self.DefineDataSet()
        
        if self.Name is None:
            self.Name = self.Code
        if self.DownloadURL is not None:
            sDownloadedFileName = os.path.basename(self.DownloadURL)
            self.ZipName, sZipExtension = os.path.splitext(sDownloadedFileName)
        else:
            self.ZipName = None
        
        
        sDataSetFolder = os.path.join("MLData", self.Code)
        self.FileStore = CFileStore(sDataSetFolder, p_bIsVerbose = self.IsVerbose)
        
        self.Items              = None
        self.ItemToSampleIDs    = None
        self.Users              = None
        self.UserToSampleIDs   = None
        self.Ratings            = None
        self.ItemCount = None
        self.UserCount = None
        self.MaxUserID = None
        self.MaxItemID = None
        self.MinRating = None
        self.MaxRating = None
        # ................................................................
        
        # Lazy dataset initialization. Try to load the data and if not already cached to local filestore, generate the samples now and cache them.
        self.Samples            = self.FileStore.Deserialize("%s-Samples.pkl" % self.Code)
        self.Ratings            = self.FileStore.Deserialize("%s-Ratings.pkl" % self.Code)
        self.Items              = self.FileStore.Deserialize("%s-Items.pkl" % self.Code)
        self.ItemToSampleIDs    = self.FileStore.Deserialize("%s-Items-SampleIDs.pkl" % self.Code)
        self.Users              = self.FileStore.Deserialize("%s-Users.pkl" % self.Code)
        self.UserToSampleIDs   = self.FileStore.Deserialize("%s-Users-SampleIDs.pkl" % self.Code)
        if self.Samples is None:
            self.ImportData()
        
        
        self.Labels = self.Ratings
        self.PreExistingSplit()
        
        self.ItemCount   = len(self.Items.keys())
        self.MaxItemID   = self.ItemCount - 1
        
        self.UserCount = len(self.Users.keys())
        self.MaxUserID   = self.UserCount - 1
        
        self.MinRating   = np.min(self.Ratings)
        self.MaxRating   = np.max(self.Ratings)
        
        self.SampleCount = self.Samples.shape[0]
    # --------------------------------------------------------------------------------------
    def PreExistingSplit(self):        
        # Using the training ratings, and validating on the same rating for item-to-item collaborative filtering
        self.TSSamples, self.VSSamples = (self.Samples, self.Samples)
        self.TSLabels, self.VSLabels = (self.Labels, self.Labels)

        self.TSSampleCount = self.TSSamples.shape[0]
        self.VSSampleCount = self.VSSamples.shape[0]
        
        print("%d samples in the Training Set" % self.TSSampleCount)
        print("%d samples in the Validation Set"%  self.VSSampleCount)
        print('.'*80)        
    # --------------------------------------------------------------------------------------
    def DefineDataSet(self): # virtual method
        self.Name            = "Food.COM"
        self.Code            = "foodcom"
        self.RatingsFile     = "interactions_train.csv"
        self.ItemsFile       = "PP_recipes.csv"
        self.ItemsFileRaw    = "RAW_recipes.csv"
        self.UsersFile       = "PP_users.csv"
    # --------------------------------------------------------------------------------------
    def DownloadZipFile(self, p_sURL, p_sDownloadFolder):
        if self.IsVerbose:
            print("Downloading %s from %s ..." % (self.ZipName, p_sURL))
        oZipFile = ZipFile(BytesIO(requests.get(p_sURL).content))
        if self.IsVerbose:
            print("Extracting %s." % self.ZipName)
        for sFileInArchive in oZipFile.namelist():
            oZipFile.extract(sFileInArchive, self.FileStore.BaseFolder)

      #oZipFileStream = oZipFile.open(oZipFile).read().decode('utf8')
      #return StringIO(oZipFileStream)
    # --------------------------------------------------------------------------------------
    def DownloadData(self):
        if self.ZipName is not None:
            sDataFileName = os.path.join(self.FileStore.BaseFolder,os.path.join(self.ZipName, self.RatingsFile))
            if not os.path.isfile(sDataFileName):
              self.DownloadZipFile(self.DownloadURL, self.FileStore.BaseFolder)
    # --------------------------------------------------------------------------------------
    def CreateItemsDict(self, p_nIDColIndex=0): # override
        if self.Items is not None:
            return

        dDataFrame          = self.ImportCSVData(self.ItemsFile)
        self.ItemToSampleIDs = dict()
        nTotal = dDataFrame.shape[0]
        for nIndex, oItem in dDataFrame.iterrows():
            if nIndex % 25000 == 0:
                print("Creating ItemID to ItemSampleID dictionary:%d/%d" % (nIndex, nTotal))
            self.ItemToSampleIDs[int(oItem[0])] = int(oItem[1])
        print("Creating ItemID to ItemSampleID dictionary:%d/%d" % (nIndex, nTotal))
        
        
        self.Items = dict()
        dDataFrameOriginal  = self.ImportCSVData(self.ItemsFileRaw)
        nMinID = 0
        nTotal = dDataFrameOriginal.shape[0]
        for nIndex, oItem in dDataFrameOriginal.iterrows():
            if nIndex % 25000 == 0:
                print("Creating items cache:%d/%d" % (nIndex, nTotal))
            if int(oItem[p_nIDColIndex]) == 0:
                print("Has item with ID #0")
            nItemIDOriginal = int(oItem[p_nIDColIndex])
            if nItemIDOriginal in self.ItemToSampleIDs:
                nItemID         = self.ItemToSampleIDs[nItemIDOriginal]
                oFeatures = list(oItem)
                self.Items[nItemID] = [nItemIDOriginal] + oFeatures[:p_nIDColIndex] + oFeatures[p_nIDColIndex+1:]

        
                    
        print("Creating items cache:%d/%d" % (nIndex, nTotal))
        print("Saving to cache ...")
        self.FileStore.Serialize("%s-Items.pkl" % self.Code, self.Items)
        self.FileStore.Serialize("%s-Items-SampleIDs.pkl" % self.Code, self.ItemToSampleIDs)

        
    # --------------------------------------------------------------------------------------
    def CreateUsersDict(self, p_nIDColIndex=0):  # override
        if self.Users is not None:
            return 
        
        self.Users = dict()
        dDataFrame = self.ImportCSVData(self.UsersFile)   
        nTotal = dDataFrame.shape[0]  
        for nIndex, oUser in dDataFrame.iterrows():
            if nIndex % 25000 == 0:
                print("Creating users cache:%d/%d" % (nIndex, nTotal))
            
            nUserID = int(oUser[p_nIDColIndex])
            oFeatures = list(oUser)
            self.Users[nUserID] = oFeatures[:p_nIDColIndex] + oFeatures[p_nIDColIndex+1:]
            
        print("Creating users cache:%d/%d" % (nIndex, nTotal))
        print("Saving to cache ...")
        self.FileStore.Serialize("%s-Users.pkl" % self.Code, self.Users)
                
    # --------------------------------------------------------------------------------------
    def CreateUserItemRatings(self, p_nRatingsColIndex=3): # override
        if self.Samples is not None:
            return 
        
        dDataFrame = self.ImportCSVData(self.RatingsFile);
    
        nUserItemCombinations   = dDataFrame.iloc[:, :2].values 
        nUserItemRatings        = dDataFrame.iloc[:, p_nRatingsColIndex].values.astype(np.float32)
    
        self.Samples     = nUserItemCombinations
        self.Ratings     = nUserItemRatings
        # Serializes all objects that the previous methods have created into pickle files
        self.FileStore.Serialize("%s-Samples.pkl" % self.Code, self.Samples)
        self.FileStore.Serialize("%s-Ratings.pkl" % self.Code, self.Ratings) 
    # --------------------------------------------------------------------------------------
    def ImportCSVData(self, p_sFileName, p_sDelimiter=",", p_bHasHeader=True):
        if self.ZipName is not None:
            sFileInFolderName = os.path.join(self.ZipName, p_sFileName)
        else:
            sFileInFolderName = p_sFileName
        sFileName = os.path.join(self.FileStore.BaseFolder, sFileInFolderName)
        if self.IsVerbose:
            print("Loading file %s" % sFileName)
        
        if p_bHasHeader:
            nHeader = 0
        else:
            nHeader = None
        
        return pd.read_csv(sFileName, sep=p_sDelimiter, header=nHeader, engine="python", encoding = "ISO-8859-1")
    # --------------------------------------------------------------------------------------
    def CreateContinuousUserIDs(self):
        if self.UserToSampleIDs is not None:
            return 
        
        self.UserToSampleIDs = dict()
        nNextID = 0
        nTotal = self.Samples.shape[0]  
        for nIndex, oSample in enumerate(self.Samples):
            if nIndex % 100000 == 0:
                print("Iterating on samples:%d/%d" % (nIndex, nTotal))
            nUserID = int(oSample[0])
            if nUserID not in self.UserToSampleIDs:
                self.UserToSampleIDs[nUserID] = nNextID
                nNextID += 1
                if nNextID % 5000 == 0:
                    print("  |__ Create %d unique UserSampleIDs" % nNextID)
        print("Iterating on samples:%d/%d" % (nIndex, nTotal))
        print("  |__ Found %d unique users" % nNextID)
        
            
        print("Saving to cache ...")
        self.FileStore.Serialize("%s-Users-SampleIDs.pkl" % self.Code, self.UserToSampleIDs)
    # --------------------------------------------------------------------------------------
    def ReplaceItemUserWithSamplesID(self):
        nTotal = self.Samples.shape[0]
        for nIndex, oSample in enumerate(self.Samples):
            if nIndex % 100000 == 0:
                print("Replacing IDs with SampleIDs:%d/%d" % (nIndex, nTotal))            
            self.Samples[nIndex][0] = self.UserToSampleIDs[oSample[0]]
            self.Samples[nIndex][1] = self.ItemToSampleIDs[oSample[1]]
        print("Replacing IDs with SampleIDs:%d/%d" % (nIndex, nTotal))             
        
        print("Saving to cache ...")                               
        self.FileStore.Serialize("%s-Samples.pkl" % self.Code, self.Samples, p_bIsOverwritting=True)                                    
    # --------------------------------------------------------------------------------------
    def ImportData(self):
        self.DownloadData()
        self.CreateItemsDict(p_nIDColIndex=1)
        self.CreateUsersDict(p_nIDColIndex=0)
        self.CreateUserItemRatings()
        self.CreateContinuousUserIDs() 
        self.ReplaceItemUserWithSamplesID()
    # --------------------------------------------------------------------------------------
# =========================================================================================================================


if __name__ == "__main__":
    oDataSet = CFoodcomRatingsDataset(p_bIsVerbose=True)
    print("-"*40, oDataSet.Name, "-"*40)
    #oDataSet.Split(0.9)
    print("Users:%d Items:%d Ratings:%d" % (oDataSet.UserCount, oDataSet.ItemCount, oDataSet.SampleCount))
    #print("Training Samples:"   , oDataSet.TSSamples.shape)
    #print("Validation Samples:" , oDataSet.VSSamples.shape)
    
    






