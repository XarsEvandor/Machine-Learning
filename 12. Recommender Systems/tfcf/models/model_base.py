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

import os
import inspect
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from tfcf.config import Config

#--------------------------------------------------------------------------------------------------------------
def _class_vars(obj):
    return {k: v for k, v in inspect.getmembers(obj)
            if not k.startswith('__') and not callable(k)}
#--------------------------------------------------------------------------------------------------------------




# =======================================================================================================================
class BaseModel(object):
    """Base model for SVD and SVD++.
    """
    #--------------------------------------------------------------------------------------------------------------
    def __init__(self, config=None, p_oDataSet=None):
        self._built = False
        self._saver = None
        self.Name = "RSModel"

        if (config is None):
          if p_oDataSet is not None:
              config = Config()
              config.num_users = p_oDataSet.UserCount
              config.num_items = p_oDataSet.ItemCount
              config.min_value = p_oDataSet.MinRating
              config.max_value = p_oDataSet.MaxRating
          else:
            raise Exception("You should provide count of items and users to the model")

        for attr in _class_vars(config):
            name = attr if not attr.startswith('_') else attr[1:]
            setattr(self, name, getattr(config, attr))
    #--------------------------------------------------------------------------------------------------------------
    def SaveModel(self, model_dir):
        """Saves Tensorflow model.

        Args:
            model_dir: A string, the path of saving directory
        """

        if not self._built:
            raise RunTimeError('The model must be trained '
                               'before saving.')

        self._saver = tfv1.train.Saver()

        model_name = type(self).__name__

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, model_name)

        self._saver.save(self._sess, model_path)
    #--------------------------------------------------------------------------------------------------------------
    def LoadModel(self, model_dir):
        """Loads Tensorflow model.

        Args:
            model_dir: A string, the path of saving directory
        """

        tensor_names = ['placeholder/users:0', 'placeholder/items:0',
                        'placeholder/ratings:0', 'prediction/pred:0']
        operation_names = ['optimizer/optimizer']

        model_name = type(self).__name__

        model_path = os.path.join(model_dir, model_name)

        self._saver = tfv1.train.import_meta_graph(model_path + '.meta')
        self._saver.restore(self._sess, model_path)

        for name in tensor_names:
            attr = '_' + name.split('/')[1].split(':')[0]
            setattr(self, attr, tfv1.get_default_graph().get_tensor_by_name(name))

        for name in operation_names:
            attr = '_' + name.split('/')[1].split(':')[0]
            setattr(self, attr, tfv1.get_default_graph(
            ).get_operation_by_name(name))

        self._built = True
    #--------------------------------------------------------------------------------------------------------------
# =======================================================================================================================