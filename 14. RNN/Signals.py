from mllib.signals.multisignal import CMITMultiSignalRecording
from mllib.visualization.multiseriegraph import CMultiSerieGraph


oFilesAndChannels = {"AccTempEDA": ["ax", "ay", "az", "temp", "EDA"], "SpO2HR":  ["SpO2", "hr"]}

nMaxPersonNumber = 21
    
DATASET_FOLDER = r"G:\MLDataSets.2022\Physionet-Non-EEG Dataset for Assessment of Neurological Status"

oRecording = CMITMultiSignalRecording(DATASET_FOLDER, oFilesAndChannels, p_sFileFormat="Subject%d_%s")
for nPersonNumber in range(1, nMaxPersonNumber):
  oRecording.ReadRecording(nPersonNumber)
  #nSignalIndex = 6
  #draw_ecg_R("Subject %d channel %s" % (nPersonNumber, oRecording.SignalNames[nSignalIndex]),oRecording.Signals[nSignalIndex],oRecording.Annotations)

  sTitle      = "Person %d recorded signals" % nPersonNumber
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
        