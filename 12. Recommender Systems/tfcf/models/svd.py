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
import tensorflow as tf

# [PANTELIS] If we want to port an existing model from Tensorflow V1 to the Tensorflow V2 we should use this package and the tfv1 alias for any existing declaration
import tensorflow.compat.v1 as tfv1


try:
    from tensorflow.keras import utils
except:
    from tensorflow.contrib.keras import utils

from tfcf.models.model_base import BaseModel
from tfcf.utils.data_utils import BatchGenerator
from tfcf.metrics import mae
from tfcf.metrics import rmse

from mllib.helpers import CModelConfig

# =======================================================================================================================
class SVD(BaseModel):
    """Collaborative filtering model based on SVD algorithm.
    """
    # https://github.com/WindQAQ/tf-recsys

    #--------------------------------------------------------------------------------------------------------------
    def __init__(self, sess, config=None, p_oConfig=None, p_oDataSet=None):
        super(SVD, self).__init__(config=config, p_oDataSet=p_oDataSet)
        self.Name = "SVD"
        self._sess = sess
        self.Config = CModelConfig(self, p_oConfig)
    #--------------------------------------------------------------------------------------------------------------
    def _create_placeholders(self):
        """Returns the placeholders.
        """
        with tfv1.variable_scope('placeholder'):
            users   = tfv1.placeholder(tf.int32, shape=[None, ], name='users')
            items   = tfv1.placeholder(tf.int32, shape=[None, ], name='items')
            ratings = tfv1.placeholder(tf.float32, shape=[None, ], name='ratings')

        return users, items, ratings
    #--------------------------------------------------------------------------------------------------------------
    def _create_constants(self, mu):
        """Returns the constants.
        """
        with tfv1.variable_scope('constant'):
            _mu = tfv1.constant(mu, shape=[], dtype=tf.float32)

        return _mu
    #--------------------------------------------------------------------------------------------------------------
    def _create_user_terms(self, users):
        """Returns the tensors related to users.
        """
        num_users = self.num_users
        num_factors = self.num_factors

        with tfv1.variable_scope('user'):
            user_embeddings = tfv1.get_variable(
                name='embedding',
                shape=[num_users, num_factors],
                initializer=tf.keras.initializers.glorot_normal(),
                regularizer=tf.keras.regularizers.l2(self.reg_p_u))

            user_bias = tfv1.get_variable(
                name='bias',
                shape=[num_users, ],
                initializer=tf.keras.initializers.glorot_normal(),
                regularizer=tf.keras.regularizers.l2(self.reg_b_u))

            p_u = tfv1.nn.embedding_lookup(
                user_embeddings,
                users,
                name='p_u')

            b_u = tfv1.nn.embedding_lookup(
                user_bias,
                users,
                name='b_u')

        return p_u, b_u
    #--------------------------------------------------------------------------------------------------------------
    def _create_item_terms(self, items):
        """Returns the tensors related to items.
        """
        num_items = self.num_items
        num_factors = self.num_factors

        with tfv1.variable_scope('item'):
            item_embeddings = tfv1.get_variable(
                name='embedding',
                shape=[num_items, num_factors],
                initializer=tf.keras.initializers.glorot_normal(),
                regularizer=tf.keras.regularizers.l2(self.reg_q_i))

            item_bias = tfv1.get_variable(
                name='bias',
                shape=[num_items, ],
                initializer=tf.keras.initializers.glorot_normal(),
                regularizer=tf.keras.regularizers.l2(self.reg_b_i))

            q_i = tfv1.nn.embedding_lookup(
                item_embeddings,
                items,
                name='q_i')

            b_i = tfv1.nn.embedding_lookup(
                item_bias,
                items,
                name='b_i')

        return q_i, b_i
    #--------------------------------------------------------------------------------------------------------------
    def _create_prediction(self, mu, b_u, b_i, p_u, q_i):
        """Returns the tensor of prediction.

           Note that the prediction 
            r_hat = \mu + b_u + b_i + p_u * q_i
        """
        with tfv1.variable_scope('prediction'):
            pred = tfv1.reduce_sum( tfv1.multiply(p_u, q_i), axis=1)

            pred = tfv1.add_n([b_u, b_i, pred])

            pred = tfv1.add(pred, mu, name='pred')

        return pred
    #--------------------------------------------------------------------------------------------------------------
    def _create_loss(self, pred, ratings):
        """Returns the L2 loss of the difference between
            ground truths and predictions.

           The formula is here:
            L2 = sum((r - r_hat) ** 2) / 2
        """
        with tfv1.variable_scope('loss'):
            loss = tfv1.nn.l2_loss(tf.subtract(ratings, pred), name='loss')

        return loss
    #--------------------------------------------------------------------------------------------------------------
    def _create_optimizer(self, tLoss):
        """Returns the optimizer.

           The objective function is defined as the sum of
            loss and regularizers' losses.
        """
        with tfv1.variable_scope('optimizer'):
            tLambda = tfv1.constant(self.Config.Value["Training.WeightDecay"], tf.float32)
            tRegularizationLoss = tfv1.multiply( tLambda, tfv1.add_n(tfv1.get_collection(tfv1.GraphKeys.REGULARIZATION_LOSSES)) )
            tCostFunction = tfv1.add(tLoss, tRegularizationLoss, 'objective')
            
            if self.Config.Value["Training.Optimizer"] == "NADAM":
                oOptimizer =  tf.compat.v1.keras.optimizers.Nadam(self.Config.Value["Training.LearningRate"])
            elif self.Config.Value["Training.Optimizer"] == "ADAM":
                oOptimizer = tfv1.train.AdamOptimizer(self.Config.Value["Training.LearningRate"])
            elif self.Config.Value["Training.Optimizer"] == "MOMENTUM":
                oOptimizer = tfv1.train.MomentumOptimizer(self.Config.Value["Training.LearningRate"], momentum=self.Config.Value["Training.Momentum"])
            
            print("Using %s optimizer" % self.Config.Value["Training.Optimizer"])
            tMinimize = oOptimizer.minimize(tCostFunction, name='optimizer')

        return tMinimize
    #--------------------------------------------------------------------------------------------------------------
    def _build_graph(self, mu):
        _mu = self._create_constants(mu)

        self._users, self._items, self._ratings = self._create_placeholders()

        p_u, b_u = self._create_user_terms(self._users)
        q_i, b_i = self._create_item_terms(self._items)

        self._pred = self._create_prediction(_mu, b_u, b_i, p_u, q_i)

        loss = self._create_loss(self._ratings, self._pred)

        self._optimizer = self._create_optimizer(loss)

        self._built = True
    #--------------------------------------------------------------------------------------------------------------
    def _run_train(self, x, y, epochs, batch_size, validation_data):
        train_gen = BatchGenerator(x, y, batch_size)
        steps_per_epoch = np.ceil(train_gen.length / batch_size).astype(int)

        self._sess.run(tfv1.global_variables_initializer())

        for e in range(1, epochs + 1):
            print('Epoch {}/{}'.format(e, epochs))

            pbar = utils.Progbar(steps_per_epoch)

            for step, batch in enumerate(train_gen.next(), 1):
                users = batch[0][:, 0]
                items = batch[0][:, 1]
                ratings = batch[1]

                self._sess.run(
                    self._optimizer,
                    feed_dict={
                        self._users: users,
                        self._items: items,
                        self._ratings: ratings
                    })

                pred = self.predict(batch[0])

                update_values = [
                    ('rmse', rmse(ratings, pred)),
                    ('mae', mae(ratings, pred))
                ]

                if validation_data is not None and step == steps_per_epoch:
                    valid_x, valid_y = validation_data
                    valid_pred = self.predict(valid_x)

                    update_values += [
                        ('val_rmse', rmse(valid_y, valid_pred)),
                        ('val_mae', mae(valid_y, valid_pred))
                    ]

                pbar.update(step, values=update_values, finalize=(step == steps_per_epoch))
    #--------------------------------------------------------------------------------------------------------------
    def train(self, x, y, epochs=100, batch_size=1024, validation_data=None, p_nMeanRating=None):

        if x.shape[0] != y.shape[0] or x.shape[1] != 2:
            raise ValueError('The shape of x should be (samples, 2) and '
                             'the shape of y should be (samples, 1).')

        if not self._built:
            if p_nMeanRating is None:
                p_nMeanRating = np.mean(y)
            self._build_graph(p_nMeanRating)

        self._run_train(x, y, epochs, batch_size, validation_data)
    #--------------------------------------------------------------------------------------------------------------
    def predict(self, x):
        if not self._built:
            raise RunTimeError('The model must be trained '
                               'before prediction.')

        if x.shape[1] != 2:
            raise ValueError('The shape of x should be '
                             '(samples, 2)')

        pred = self._sess.run(
            self._pred,
            feed_dict={
                self._users: x[:, 0],
                self._items: x[:, 1]
            })

        pred = pred.clip(min=self.min_value, max=self.max_value)

        return pred
    #--------------------------------------------------------------------------------------------------------------
# =======================================================================================================================