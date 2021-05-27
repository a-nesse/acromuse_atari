# Copyright 2021 Andreas Nesse
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers

class AtariNet(tf.keras.Sequential):
    """
    Wrapper class for creating Keras network used for playing Atari games.
    """


    def __init__(self, input_shape, action_shape, net_conf, minval=-1, maxval=1, val_buffer=1e-6):

        super().__init__()

        self.minval = minval
        self.maxval = maxval
        self.replace_min = minval+val_buffer

        #initialize weights
        initializer = initializers.RandomUniform(minval=minval, maxval=maxval)
        
        self.action_shape = action_shape

        self.add(keras.Input(shape=input_shape))
        for conv in net_conf['conv_layer_params']:
            self.add(layers.Conv2D(
                conv[0],
                conv[1],
                conv[2],
                activation=net_conf['conv_activation'],
                kernel_initializer=initializer,
                bias_initializer=initializer))
        self.add(layers.Flatten())
        for full in net_conf['fc_layer_params']:
            self.add(layers.Dense(
                full,
                activation=net_conf['fc_activation'],
                kernel_initializer=initializer,
                bias_initializer=initializer))
        self.add(layers.Dense(
            action_shape,
            activation=None,
            kernel_initializer=initializer,
            bias_initializer=initializer))
        self.build()


    def get_weights(self):
        """
        Gets weights and converts to an array of np.array objects.
        """
        return np.array(super().get_weights(),dtype=object)


    def get_scaled_weights(self):
        """
        Returns weights shifted & scaled to the range <0,1> using the specified minimum & maximum weight value.
        """
        span = self.maxval-self.minval
        return (self.get_weights()-self.minval)/span


    def set_weights(self,weights):
        """
        Receives an array of np.arrays, converts to a list to set as weights for the agent.
        """
        for i,layer in enumerate(weights):
            #checking for any values equal to minval
            if np.any(layer==self.minval):
                weights[i]=np.where(weights[i]==self.minval,self.replace_min,weights[i])
        super().set_weights(list(weights))


    def action(self, observation, epsilon=0):
        """
        Returns action with highest activation in output layer.
        """
        if epsilon and epsilon>np.random.rand():
            return np.random.randint(self.action_shape)
        activations = super().predict(observation.observation)
        return np.argmax(activations)