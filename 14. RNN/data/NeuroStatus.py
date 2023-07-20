import os
import numpy as np
import random

if __name__ == "__main__":
	import sys
	sys.path.append(".")
	sys.path.append("..")
	os.chdir("..")


from mllib.data import CCustomDataSet
from mllib.filestore import CFileStore
from mllib.signals.multisignal import CMITMultiSignalRecording



# =========================================================================================================================
class CNeuroStatus(CCustomDataSet):
	# ------------------------------------------------------------------------------------
	def __init__(self, p_sName=None, p_bIsVerbose=False):
		super(CNeuroStatus, self).__init__()
		# ................................................................
		# // Fields \\
		self.IsVerbose = p_bIsVerbose
		if p_sName is None:
			p_sName = "NeuroStatusSignals"
		self.Name      = p_sName
		self.FileStore = CFileStore(os.path.join("MLData", self.Name), p_bIsVerbose = self.IsVerbose)
		self.PersonCount = 20
		self.PersonSamples = None
		self.PersonLabels  = None
		
		# ................................................................
		
		# Lazy dataset initialization. Try to load the data and if not already cached to local filestore, generate the samples now and cache them.
		self.Samples						= self.FileStore.Deserialize("%s-Samples.pkl" % self.Name)
		self.Labels 						= self.FileStore.Deserialize("%s-Labels.pkl" % self.Name)
		if self.Samples is None:
			self.BuildDataSet()
			self.FileStore.Serialize("%s-Samples.pkl" % self.Name, self.Samples)
			self.FileStore.Serialize("%s-Labels.pkl" % self.Name, self.Labels)
		
		self.countSamples()
	# --------------------------------------------------------------------------------------
	def BuildDataSet(self):
		self.LoadPersonSamples()
	# --------------------------------------------------------------------------------------
	def LoadPersonSamples(self):
		oFilesAndChannels = {"AccTempEDA":["ax","ay","az","temp","EDA"], "SpO2HR":["SpO2", "hr"]}
		sSourceFolder = os.path.join(self.FileStore.BaseFolder, "signals")
		oMultiSignal = CMITMultiSignalRecording(sSourceFolder, oFilesAndChannels, p_sFileFormat="Subject%d_%s")
		for nPersonNumber in range(1, self.PersonCount + 1):
			oMultiSignal.ReadRecording(nPersonNumber)	
		
			if self.PersonSamples is None:
				nFeatureCount   = len(oMultiSignal.SignalNames)
				self.PersonSamples = np.zeros((self.PersonCount, oMultiSignal.DataPointCount, nFeatureCount), np.float32)
				self.PersonLabels	 = np.zeros((self.PersonCount), np.int32)
				self.Samples = []
				self.Labels = []
				print("The shape of the person sample set is", self.PersonSamples.shape)
			nPersonIndex = nPersonNumber - 1
			self.PersonLabels[nPersonIndex] = nPersonIndex
			for nIndex,nSignal in enumerate(oMultiSignal.Signals):
				print(nIndex, nSignal.shape)
				if nSignal.ndim == 2:
					self.PersonSamples[nPersonIndex,:,nIndex] = nSignal[:,0]
				else:
					self.PersonSamples[nPersonIndex,:,nIndex] = nSignal[:]
			
			self.CreateMovingWindowSamples(nPersonIndex)
			
		self.Samples = np.array(self.Samples, np.float32)
		self.Labels  = np.array(self.Labels, np.int32)
	# --------------------------------------------------------------------------------------
	def CreateMovingWindowSamples(self, p_nPersonIndex, p_nWindowSize=49, p_nStride=5):
		nPersonSample = self.PersonSamples[p_nPersonIndex, ...]
		nLabel = self.PersonLabels[p_nPersonIndex]
		nIndex = 0
		nMaxIndex = self.PersonSamples.shape[1]
		nSeqIndex = 0
		oSeq = []
		while nIndex < nMaxIndex:
			if (nIndex*p_nWindowSize + p_nWindowSize) < nMaxIndex:
				print("Seq#%d window from:to %d:%d" % ((nSeqIndex+1), nIndex*p_nWindowSize, nIndex*p_nWindowSize + p_nWindowSize))
				nSample = nPersonSample[nIndex*p_nWindowSize: nIndex*p_nWindowSize + p_nWindowSize, :]
				oSeq.append(nSample)
				nSeqIndex += 1
				
			nIndex += p_nStride
		self.Samples.append(oSeq)
		self.Labels.append(nLabel)
	# --------------------------------------------------------------------------------------
	def SplitSequences(self, p_nSplitClips=25):
		nShape = list(self.Samples.shape)
		nShape[1] = nShape[1] - p_nSplitClips
		self.TSSamples = np.zeros(nShape, np.float32)
		self.TSLabels  = np.zeros(nShape[0], np.int32)
		
		self.VSSamples = np.zeros(nShape, np.float32)
		self.VSLabels  = np.zeros(nShape[0], np.int32)
		
		for nIndex, nSequence in enumerate(self.Samples):
			nVSClipCount = 0
			nTSClipCount = 0
			nLabel = self.Labels[nIndex]
			for nClipIndex, nClip in enumerate(nSequence):
				bIsValidation = False
				if (nClipIndex % 2) == 0:
					if nVSClipCount < p_nSplitClips:
						bIsValidation = True
				
				if bIsValidation:
					self.VSSamples[nIndex,-(p_nSplitClips-nVSClipCount),:,:] = nClip[:,:]
					self.VSLabels[nIndex] = nLabel
					nVSClipCount += 1
				else:
					self.TSSamples[nIndex,nTSClipCount,:,:] = nClip[:,:]
					self.TSLabels[nIndex] = nLabel
					nTSClipCount += 1
				
		
		self.countSamples()
	# --------------------------------------------------------------------------------------

# =========================================================================================================================		
		
		
if __name__ == "__main__":
	oDataSet = CNeuroStatus(p_bIsVerbose=True)
	oDataSet.SplitSequences(25)
	print("-"*40, oDataSet.Name, "-"*40)
		
	print("%d sequences with %d clips, each one %d data points with %d features" % oDataSet.Samples.shape)
	print("%d labels for sequences" % oDataSet.Labels.shape)
	
	print("."*20, "Training Set", "."*40)
	print("%d sequences with %d clips, each one %d data points with %d features" % oDataSet.TSSamples.shape)
	print("%d labels for sequences" % oDataSet.TSLabels.shape)
	
	print("."*20, "Validation Set", "."*40)
	print("%d sequences with %d clips, each one %d data points with %d features" % oDataSet.VSSamples.shape)
	print("%d labels for sequences" % oDataSet.VSLabels.shape)

		
	
	