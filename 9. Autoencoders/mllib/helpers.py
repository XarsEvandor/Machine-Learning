import csv


# =========================================================================================================================
class CModelConfig(object):
    # --------------------------------------------------------------------------------------
    def __init__(self, p_oModel, p_oValuesDictionary):
        self.Model = p_oModel
        self.Value = dict()
        for sKey in p_oValuesDictionary.keys():
            self.Value[sKey] = p_oValuesDictionary[sKey]
    # --------------------------------------------------------------------------------------
    def DefaultValue(self, p_sKey, p_oValue):
        if p_sKey not in self.Value:
            self.Value[p_sKey] = p_oValue        
    # --------------------------------------------------------------------------------------
# =========================================================================================================================

        


# =========================================================================================================================
class CKerasModelStructureElement(list):
    # --------------------------------------------------------------------------------------    
    # Constructor
    def __init__(self, p_sName, p_oShape):
        # ..................... Object Attributes ...........................
        self.Name  = p_sName
        self.Shape = p_oShape
        # ...................................................................
    # --------------------------------------------------------------------------------------
    def __str__(self):
        return "%64s %s" % (self.Name, self.Shape)
        
    # --------------------------------------------------------------------------------------
# =========================================================================================================================

# =========================================================================================================================
class CKerasModelStructure(list):
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self, p_bIsEnabled=True):
    # ..................... Object Attributes ...........................
    self.SoftmaxActivation  = None
    self.IsEnabled = p_bIsEnabled
    self.LayerNumber = 0
    # ...................................................................
  # --------------------------------------------------------------------------------------
  def Add(self, p_tTensor):
      if self.IsEnabled:
          self.append(CKerasModelStructureElement(p_tTensor.name, p_tTensor.shape))
  # --------------------------------------------------------------------------------------
  def Print(self, p_sWriteToFileName):
      with open(p_sWriteToFileName, 'w') as f: 
          write = csv.writer(f)
          for nIndex,oElement in enumerate(self):
              print(nIndex, oElement) 
              write.writerow("%d;%s;%s" % (nIndex+1, oElement.Name, oElement.Shape))
  # --------------------------------------------------------------------------------------        
# =========================================================================================================================