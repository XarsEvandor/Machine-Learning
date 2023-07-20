import os
import numpy as np
from scipy.interpolate import interp1d
import wfdb as wfdb

# =========================================================================================================================
class CMITMultiSignalRecording(object):
  # --------------------------------------------------------------------------------------------------------
  def __init__(self, p_sDataFolder, p_sFilesAndChannels, p_sFileFormat):
    self.DataFolder = p_sDataFolder
    self.FilesAndChannels = p_sFilesAndChannels
    self.FileFormat = p_sFileFormat
    
    self.Signals      = []  
    self.SignalNames  = []
    self.Time         = []
    self.Annotations  = None
    self.DataPointCount = None
  # --------------------------------------------------------------------------------------------------------
  def interpolateSignal(self, p_nLFSignal):
    nSignalLen = p_nLFSignal.shape[0]
    x_LF = np.linspace(0, nSignalLen, num=nSignalLen, endpoint=True)
    y_LF = p_nLFSignal.reshape(-1)
    f = interp1d(x_LF, y_LF, kind="cubic") 
  
    x_HF = np.linspace(0, nSignalLen, num=self.DataPointCount, endpoint=True)
    nInterpolatedSignal = f(x_HF)
    return nInterpolatedSignal    
  # --------------------------------------------------------------------------------------------------------
  def ReadRecording(self, p_nRecordingNumber ):
    self.Signals = []
    self.Annotations = None
    nBaseSignalLen = None
    for sFileSuffix in self.FilesAndChannels:
      oChannelNames = self.FilesAndChannels[sFileSuffix]
      
      for sChannelName in oChannelNames:
        sFileNameOnly = os.path.join(self.DataFolder, self.FileFormat % (p_nRecordingNumber, sFileSuffix))
        oChannelRecord = wfdb.rdrecord(sFileNameOnly,channel_names=[sChannelName])
        nSignal = oChannelRecord.p_signal
        nSignalLen = nSignal.shape[0]
        
        if self.DataPointCount is None:
          self.DataPointCount = nSignalLen
          self.Time = np.linspace(0, self.DataPointCount, num=self.DataPointCount, endpoint=True)
        
        if nSignalLen != self.DataPointCount:
          nSignal = self.interpolateSignal(nSignal)       

        
        self.Signals.append(nSignal)
        self.SignalNames.append(sChannelName)
         
        if (self.Annotations is None):
          if os.path.isfile(sFileNameOnly + ".atr"):
            oAnnotationRecord = wfdb.rdann(sFileNameOnly,'atr')
            self.Annotations = oAnnotationRecord.sample
  # --------------------------------------------------------------------------------------------------------            
# =========================================================================================================================    
  
  