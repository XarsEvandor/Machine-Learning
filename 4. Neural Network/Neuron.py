import numpy as np
import matplotlib.pyplot as plt

# ====================================================================================================
class CNeuron(object):
  IS_USING_LINEAR_ALGEBRA = True

    
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self, p_nDendriteCount, p_sActivationFunction):
    # ................................................................
    self.DendriteCount = p_nDendriteCount
    self.ActivationFunction = p_sActivationFunction
    
    self.Input          = None
    self.Output         = None
    self.weights        = np.random.randn((self.DendriteCount)).astype(np.float64)  # Random initialization of numpy array
    self.bias           = 0.0
    # ................................................................
  # --------------------------------------------------------------------------------------
  def stepActivationFunction(self, u):  
    if u >= 0:
        return 1.0  # fires 1, when the amount of energy surpases the threshold 
    else:
        return 0.0  # silent
  # --------------------------------------------------------------------------------------
  def stepOneMinusOneActivationFunction(self, u):  
    if u >= 0:
        return 1.0  # fires an excitatory signal 1 (positive)
    else:
        return -1.0 # fires an inhibitory signal -1 (negative)
  # --------------------------------------------------------------------------------------
  def sigmoidActivationFunction(self, u):
    y = 1/(1 + np.exp(-u)) 
    return y
  # --------------------------------------------------------------------------------------
  def rectifiedLinearUnitActivationFunction(self, u):
    if u >= 0:
        return u  # fires the exact amount of incoming energy, when the amount of energy surpases the threshold 
    else:
        return 0.0  # silent
    return y
  # --------------------------------------------------------------------------------------
  # [PYTHON] There are only private and public visibilities in Python.
  #          When you put a double underscore as a prefix of member name it becomes private.     
  def __synapticIntegration(self):
    nSum = 0.0;
    for i in range(0, self.Input.shape[0]):
        nSum += self.weights[i]*self.Input[i]

    return nSum
  # --------------------------------------------------------------------------------------
  def Recall(self, x):
    self.Input = x

    if CPerceptron.IS_USING_LINEAR_ALGEBRA:
        u = np.dot(self.weights, self.Input)
    else:
        u = self.__synapticIntegration()
    
    if self.ActivationFunction == "linear":
        a = u
    elif self.ActivationFunction == "binarystep":
        a = self.stepActivationFunction(u + self.bias)
    elif self.ActivationFunction == "step":
        a = self.stepOneMinusOneActivationFunction(u + self.bias)
    elif self.ActivationFunction == "tanh":
        a = np.tanh(u - self.bias)
    elif self.ActivationFunction == "sigmoid":
        a = self.sigmoidActivationFunction(u + self.bias)
    elif self.ActivationFunction == "relu":
        a = self.rectifiedLinearUnitActivationFunction(u + self.bias)    

    self.Output = a
    return a;
  # --------------------------------------------------------------------------------------
  def TrainPerceptron(self, p_nLearningRate, p_nError):
    # Each weight is modified by an amount that is proportional to the error
    for i in range(0, self.DendriteCount):
        nPreviousWeight = self.weights[i]
        self.weights[i] = nPreviousWeight + p_nLearningRate*p_nError*self.Input[i]
  # --------------------------------------------------------------------------------------        
  def TrainGradientDescent(self, p_nLearningRate, p_nDelta):
    for i in range(0, self.DendriteCount):
        nPreviousWeight = self.weights[i]
        self.weights[i] = nPreviousWeight - p_nLearningRate*p_nDelta*self.Input[i]   
        
    #self.bias = self.bias + p_nLearningRate*p_nDelta*(-1)   
  # --------------------------------------------------------------------------------------
  
  
# ====================================================================================================
class CPerceptron(CNeuron):
  # --------------------------------------------------------------------------------------
  # Constructor
  def __init__(self, p_nDendriteCount):
    super(CPerceptron, self).__init__(p_nDendriteCount, "binarystep")
# ====================================================================================================  


# This checks if the current python file is executing as the main method of the application
if __name__ == "__main__":


  oNeuron = CPerceptron(2)



  nXSeries = np.linspace(-10, 10, 100, dtype=np.float64)   # Create 100 real values in the space [-10.0, 10.0]


  plt.title("Linear y=x function")  
  y1 = oNeuron.linearActivationFunction(nXSeries)
  plt.plot(nXSeries, y1) 
  plt.xlabel("x") 
  plt.ylabel("y") 
  plt.show() 

  plt.title("Binary step function")  
  y2 = np.zeros(100, np.float64)
  for nIndex, x in enumerate(nXSeries):   # [PYTHON] Enumerate each single value and get this in pair with its index
    y2[nIndex] = oNeuron.stepActivationFunction(x)
  plt.plot(nXSeries, y2) 
  plt.xlabel("x") 
  plt.ylabel("y") 
  plt.show() 
  
  plt.title("Step one minus one function")  
  y3 = np.zeros(100, np.float64)
  for nIndex, x in enumerate(nXSeries):   # [PYTHON] Enumerate each single value and get this in pair with its index
    y3[nIndex] = oNeuron.stepOneMinusOneActivationFunction(x)
  plt.plot(nXSeries, y3) 
  plt.xlabel("x") 
  plt.ylabel("y") 
  plt.show() 




  plt.title("Sigmoid function")  
  y4 = oNeuron.sigmoidActivationFunction(nXSeries)
  plt.plot(nXSeries, y4) 
  plt.xlabel("x") 
  plt.ylabel("y") 
  plt.show() 
    
  plt.title("Tanh function")  
  y4 = np.tanh(nXSeries)
  plt.plot(nXSeries, y4) 
  plt.xlabel("x") 
  plt.ylabel("y") 
  plt.show() 
    
    
  plt.title("ReLU function")  
  y5 = np.zeros(100, np.float64)
  for nIndex, x in enumerate(nXSeries):   # [PYTHON] Enumerate each single value and get this in pair with its index
    y5[nIndex] = oNeuron.rectifiedLinearUnitActivationFunction(x)  
  plt.plot(nXSeries, y5) 
  plt.xlabel("x") 
  plt.ylabel("y") 
  plt.show() 
      
