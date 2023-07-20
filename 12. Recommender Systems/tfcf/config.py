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


import numpy as np

# =======================================================================================================================
class Config(object): # [PANTELIS] [PYTHON] The author added static attributes to this class. So num_users will be the same for all instances of :Config

    """Configuration class for collaborative filtering model
    """

    num_users = None
    num_items = None
    num_factors = 15

    # minimum and maximum value of prediction for clipping
    min_value = -np.inf
    max_value = np.inf

    # regularization scale
    reg_b_u = 0.0001
    reg_b_i = 0.0001
    reg_p_u = 0.005
    reg_q_i = 0.005
    reg_y_u = 0.005
    reg_g_i = 0.005
# =======================================================================================================================