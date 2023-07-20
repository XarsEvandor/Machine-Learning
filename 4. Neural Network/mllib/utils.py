import os
import random
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def RandomSeed(p_nSeed=2021):
    random.seed(p_nSeed)
    os.environ['PYTHONHASHSEED'] = str(p_nSeed)
    np.random.seed(p_nSeed)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
