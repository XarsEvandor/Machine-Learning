#	  MIT License
#    
#    Copyright (c) 2017 WindQAQ
#    
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#    
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#    
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.
#    

"""Operations related to evaluating predictions.
"""

import numpy as np

#[PYTHON] Some examples of writing a custom metric for your needs using numpy

#--------------------------------------------------------------------------------------------------------------
def mse(y, y_pred):
    """Returns the mean squared error between
    ground truths and predictions.
    """
    return np.mean((y - y_pred) ** 2)
#--------------------------------------------------------------------------------------------------------------
def rmse(y, y_pred):
    """Returns the root mean squared error between
    ground truths and predictions.
    """
    return np.sqrt(mse(y, y_pred))
#--------------------------------------------------------------------------------------------------------------
def mae(y, y_pred):
    """Returns mean absolute error between
    ground truths and predictions.
    """
    return np.mean(np.fabs(y - y_pred))
#--------------------------------------------------------------------------------------------------------------