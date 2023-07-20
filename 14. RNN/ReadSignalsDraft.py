import wfdb as wfdb
import matplotlib.pyplot as plt
import numpy as np

# Draw ECG
def draw_ecg(x):
    plt.plot(x)
    plt.title('Raw_ECG')
    plt.show()
    
#Draw the ECG and its R wave position
def draw_ecg_R(p_sTitle, p_signal,annotation,record=None):
    if record is not None:
      p_signal = record.p_signal
    plt.plot(p_signal) #Draw the ECG signal
    if annotation is not None:
      R_v=p_signal[annotation.sample] #Get R wave peak value
      for nIndex, nAnnotationSample in enumerate(annotation.sample):
        if annotation.num[nIndex] == 0:
          sColor = "r"
        else:
          sColor = "b"
        #plt.plot(annotation.sample,R_v, 'o' + sColor)#Draw R wave
        plt.plot(nAnnotationSample,R_v[nIndex],'o' + sColor)#Draw R wave
    plt.title(p_sTitle)
    plt.show()
def selData(record,annotation,label,R_left):
    a=annotation.symbol
    f=[k for k in range(len(a)) if a[k]==label] #Find the corresponding label R wave position index
    signal=record.p_signal
    R_pos=annotation.sample[f]
    res=[]
    for i in range(len(f)):
        if(R_pos[i]-R_left>0):
            res.append(signal[R_pos[i]-R_left:R_pos[i]-R_left+250])
    return res
        
# Read ECG data
def read_ecg_data(filePath,channel_names, p_bIsReadingAnnotations=True, p_bIsVerbose=False):
    '''
    Read ECG file
    sampfrom: Set the starting position for reading the ECG signal, sampfrom=0 means to start reading from 0, and the default starts from 0
    sampto: Set the end position of reading the ECG signal, sampto = 1500 means the end from 1500, the default is to read to the end of the file
    channel_names: set the name of reading ECG signal, it must be a list, channel_names=['MLII'] means reading MLII lead
    channels: Set the number of ECG signals to be read. It must be a list. Channels=[0, 3] means to read the 0th and 3rd signals. Note that the number of signals is uncertain 
    record = wfdb.rdrecord('../ecg_data/102', sampfrom=0, sampto = 1500) # read all channel signals
    record = wfdb.rdrecord('../ecg_data/203', sampfrom=0, sampto = 1500,channel_names=['MLII']) # Only read "MLII" signal
    record = wfdb.rdrecord('../ecg_data/101', sampfrom=0, sampto=3500, channels=[0]) # Only read the 0th signal (MLII)
    print(type(record)) # View record type
    print(dir(record)) # View methods and attributes in the class
    print(record.p_signal) # Obtain the ECG lead signal, this article obtains MLII and V1 signal data
    print(record.n_sig) # View the number of lead lines
    print(record.sig_name) # View the signal name (list), the lead name of this text ['MLII','V1']
    print(record.fs) # View the adoption rate
    '''
    
    record = wfdb.rdrecord(filePath,channel_names=[channel_names])
    if p_bIsVerbose:
      print('Number of lead lines:')
      print(record.n_sig) # View the number of lead lines
      print('Signal name (list)')
      print(record.sig_name) # View the signal name (list), the lead name of this text ['MLII','V1']

    '''
    Read annotation file
    sampfrom: Set the starting position for reading the ECG signal, sampfrom=0 means to start reading from 0, and the default starts from 0
    sampto: Set the end position of reading the ECG signal, sampto = 1500 means the end from 1500, the default is to read to the end of the file
    print(type(annotation)) # View the annotation type
    print(dir(annotation))# View methods and attributes in the class
    print(annotation.sample) # Mark the sharp position of the R wave of each heartbeat, corresponding to the ECG signal
    annotation.symbol #Mark the type of each heartbeat N, L, R, etc.
    print(annotation.ann_len) # The number of labels
    print(annotation.record_name) # The file name to be marked
    print(wfdb.show_ann_labels()) # View the type of heartbeat
    '''
    if p_bIsReadingAnnotations:
      annotation = wfdb.rdann(filePath,'atr')
    else:
      annotation = None
# print(annotation.symbol)
    return record,annotation

if __name__ == "__main__":
  from mllib.signals.multisignal import CMITMultiSignalRecording
  from mllib.visualization.multiseriegraph import CMultiSerieGraph
  oFilesAndChannels = {"AccTempEDA":["ax","ay","az","temp","EDA"], "SpO2HR":["SpO2", "hr"]}
  
  nMaxPersonNumber = 21
      
  DATASET_FOLDER = r"G:\MLDataSets.2022\Physionet-Non-EEG Dataset for Assessment of Neurological Status"
  
  oRecording = CMITMultiSignalRecording(DATASET_FOLDER, oFilesAndChannels, p_sFileFormat="Subject%d_%s")
  for nPersonNumber in range(1, nMaxPersonNumber):
    oRecording.ReadRecording(nPersonNumber)
    #nSignalIndex = 6
    #draw_ecg_R("Subject %d channel %s" % (nPersonNumber, oRecording.SignalNames[nSignalIndex]),oRecording.Signals[nSignalIndex],oRecording.Annotations)
  
  sTitle      = "Signals"
  sCaptionX   = "Time"
  sCaptionY   = "Value"
  
  oGraph = CMultiSerieGraph()
  oGraph.Setup.LegendFontSize=10
  oGraph.Setup.Title              = sTitle
  oGraph.Setup.CaptionX           = sCaptionX
  oGraph.Setup.CaptionY           = sCaptionY
  oGraph.Setup.CommonLineWidth    = 1.5
  oGraph.Setup.DisplayFinalValue  = True
  
  oSignalsToPlot = []
  oLabels = []
  for nSignalIndex in [4,5,6]:
    oSignalsToPlot.append(oRecording.Signals[nSignalIndex])
    oLabels.append(oRecording.SignalNames[nSignalIndex])
  
  oGraph.Initialize(oRecording.Time, oSignalsToPlot, oLabels, p_oPointsOfInterest=oRecording.Annotations)
  oGraph.Render(p_bIsMinMaxNormalized=True)
  oGraph.Plot()
        
  
if False:  
  import os
  IS_PLOTTING = True
  
  
  
  
  
  sCurrentChannel = "EDA"
  if sCurrentChannel in oFilesAndChannels["AccTempEDA"]:
    sCurrentFileSuffix = "AccTempEDA"
  else:
    sCurrentFileSuffix  = "SpO2HR"
    
  nMaxPersonNumber = 21
      
  DATASET_FOLDER = r"G:\MLDataSets.2022\Physionet-Non-EEG Dataset for Assessment of Neurological Status"
  
  for nPersonNumber in range(1, nMaxPersonNumber):
    SUBJECT_SIGNAL = "Subject%d_%s" % (nPersonNumber, sCurrentFileSuffix)
    
    sFileNameOnly = os.path.join(DATASET_FOLDER, SUBJECT_SIGNAL)
    record,annotation=read_ecg_data(sFileNameOnly, sCurrentChannel, sCurrentChannel in oFilesAndChannels["AccTempEDA"])
    # draw_ecg(record.p_signal)
    print(record.sig_name)
    print(sFileNameOnly, record.p_signal.shape)
  
    
    if IS_PLOTTING:
      draw_ecg_R("Subject %d channel %s" % (nPersonNumber, sCurrentChannel),record,annotation)
      if False:
        res=selData(record,annotation,'N',100)
        print(len(res))
        plt.plot(res[20])
        plt.show()