import os
import numpy as np
import pandas as pd
import requests

from io import BytesIO
from io import StringIO
from zipfile import ZipFile

from mllib.data import CCustomDataSet
from mllib.filestore import CFileStore


# =========================================================================================================================
class CMovieLens100K(CCustomDataSet):
  # --------------------------------------------------------------------------------------
  def __init__(self, p_bIsVerbose=False):
    super(CMovieLens100K, self).__init__()
    
    self.DefineDataSet()

    # ................................................................
    # // Fields \\
    sDownloadedFileName = os.path.basename(self.DownloadURL)
    self.ZipName, sZipExtension = os.path.splitext(sDownloadedFileName)
    
    self.IsVerbose = p_bIsVerbose

    sDataSetFolder = os.path.join("MLData", self.Code)
    self.FileStore = CFileStore(sDataSetFolder, p_bIsVerbose = self.IsVerbose)

    self.Items     = None
    self.Users     = None      
    self.Ratings   = None
    self.ItemCount = None
    self.UserCount = None
    self.MaxUserID = None
    self.MaxItemID = None
    self.MinRating = None
    self.MaxRating = None
    # ................................................................

    # Lazy dataset initialization. Try to load the data and if not already cached to local filestore, generate the samples now and cache them.
    self.Samples  = self.FileStore.Deserialize("%s-Samples-UserItemPairs.pkl" % self.Code)
    self.Ratings  = self.FileStore.Deserialize("%s-Ratings.pkl" % self.Code)
    if self.Samples is None:
      self.DownloadData()
      self.FileStore.Serialize("%s-Samples.pkl" % self.Code, self.Samples)
      self.FileStore.Serialize("%s-Ratings.pkl" % self.Code, self.Ratings)

    self.Items    = self.FileStore.Deserialize("%s-Items.pkl" % self.Code) 
    if self.Items is None:
      self.CreateItemsDict()

    self.Users    = self.FileStore.Deserialize("%s-Users.pkl" % self.Code)
    if self.Users is None:
      self.CreateUsersDict()
    
    self.Labels = self.Ratings
    self.MaxUserID   = np.max(self.Samples[:, 0]) + 1
    self.MaxItemID   = np.max(self.Samples[:, 1]) + 1
    
        
    if self.Items is not None:
      self.ItemCount   = len(self.Items.keys())
    else:
      self.ItemCount   = self.MaxItemID
        
    if self.Users is not None:
      self.UserCount = len(self.Users.keys())
    else:    
      self.UserCount   = self.MaxUserID
    
    self.MinRating   = np.min(self.Ratings)
    self.MaxRating   = np.max(self.Ratings)

    self.SampleCount = self.Samples.shape[0]
  # --------------------------------------------------------------------------------------
  def DefineDataSet(self): # virtual method
    self.Name            = "MovieLens 100K"
    self.Code            = "movielens100k"
    self.RatingsFile     = "u.data"
    self.ItemsFile       = None
    self.UsersFile       = None
    self.DownloadURL     = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
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
      sDataFileName = os.path.join(self.FileStore.BaseFolder,os.path.join(self.ZipName, self.RatingsFile))
      if not os.path.isfile(sDataFileName):
        self.DownloadZipFile(self.DownloadURL, self.FileStore.BaseFolder)

      dDataFrame = self.LoadRatingsFile();

      nUserItemCombinations   = dDataFrame.iloc[:, :2].values 
      nUserItemRatings        = dDataFrame.iloc[:, 2].values.astype(np.float32)

      self.Samples     = nUserItemCombinations
      self.Ratings     = nUserItemRatings
  # --------------------------------------------------------------------------------------
  def CreateItemsDict(self): # virtual method
    pass
  # --------------------------------------------------------------------------------------
  def CreateUsersDict(self):  # virtual method
    pass      
  # --------------------------------------------------------------------------------------
  def LoadRatingsFile(self):
    sFileName = os.path.join(self.FileStore.BaseFolder,os.path.join(self.ZipName, self.RatingsFile))
    if self.IsVerbose:
        print("Loading ratings files %s" % sFileName)
    return pd.read_csv(sFileName, sep="\t", header=None) # [PYTHON] Read a  tab separated .csv file with pandas
  # --------------------------------------------------------------------------------------
# =========================================================================================================================











# =========================================================================================================================
class CMovieLens1M(CMovieLens100K):
  # --------------------------------------------------------------------------------------
  def __init__(self, p_bIsVerbose=False):
    super(CMovieLens1M, self).__init__(p_bIsVerbose=p_bIsVerbose)
    # ................................................................
  # --------------------------------------------------------------------------------------
  def DefineDataSet(self): # override
    self.Name            = "MovieLens 1M"
    self.Code            = "movielens1m"
    self.RatingsFile     = "ratings.dat"
    self.ItemsFile       = "movies.dat"
    self.UsersFile       = "users.dat"
    self.DownloadURL     = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
  # --------------------------------------------------------------------------------------
  def CreateItemsDict(self): # override
    self.Items = dict()
    dData = self.LoadFile(self.ItemsFile)
    for nIndex, oItem in dData.iterrows():
      nItemID = int(oItem[0])
      sItemName = oItem[1]
      self.Items[nItemID] = sItemName
  # --------------------------------------------------------------------------------------
  def CreateUsersDict(self):  # override
    self.Users = dict()
    dData = self.LoadFile(self.UsersFile)     
    for nIndex, oUser in dData.iterrows():
      nUserID = int(oUser[0])
      nUserGenre = oUser[1]
      self.Users[nUserID] = nUserGenre
  # --------------------------------------------------------------------------------------
  def LoadRatingsFile(self): # override
    return self.LoadFile(self.RatingsFile)
  # --------------------------------------------------------------------------------------
  def LoadFile(self, p_sFileName):
    sFileName = os.path.join(self.FileStore.BaseFolder,os.path.join(self.ZipName, p_sFileName))
    if self.IsVerbose:
        print("Loading ratings files %s" % sFileName)
    return pd.read_csv(sFileName, sep="::", header=None, engine="python", encoding = "ISO-8859-1")
  # --------------------------------------------------------------------------------------
# =========================================================================================================================








# =========================================================================================================================
class CMovieLens10M(CMovieLens100K):
  # --------------------------------------------------------------------------------------
  def __init__(self, p_bIsVerbose=False):
    super(CMovieLens10M, self).__init__(p_bIsVerbose=p_bIsVerbose)
    # ................................................................
  # --------------------------------------------------------------------------------------
  def DefineDataSet(self):
    self.Name            = "MovieLens 10M"
    self.Code            = "movielens10m"
    self.RatingsFile     = "ratings.dat"
    self.DownloadURL     = "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
  # --------------------------------------------------------------------------------------
  def LoadRatingsFile(self):
    sFileName = os.path.join(self.FileStore.BaseFolder,os.path.join(self.ZipName, self.RatingsFile))
    if self.IsVerbose:
        print("Loading ratings files %s" % sFileName)
    return pd.read_csv(sFileName, sep="::", header=None, engine="python")
  # --------------------------------------------------------------------------------------

# =========================================================================================================================






if __name__ == "__main__":
  oDataSet = CMovieLens100K(p_bIsVerbose=True)
  print("-"*40, oDataSet.Name, "-"*40)
  oDataSet.Split(0.9)
  print("Users:%d Items:%d Ratings:%d" % (oDataSet.UserCount, oDataSet.ItemCount, oDataSet.SampleCount))
  print("Training Samples:"   , oDataSet.TSSamples.shape)
  print("Validation Samples:" , oDataSet.VSSamples.shape)


  oDataSet = CMovieLens10M(p_bIsVerbose=True)
  print("-"*40, oDataSet.Name, "-"*40)
  oDataSet.Split(0.9)
  print("Users:%d Items:%d Ratings:%d" % (oDataSet.UserCount, oDataSet.ItemCount, oDataSet.SampleCount))
  print("Training Samples:"   , oDataSet.TSSamples.shape)
  print("Validation Samples:" , oDataSet.VSSamples.shape)
