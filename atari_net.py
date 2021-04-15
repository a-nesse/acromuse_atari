import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers

class AtariNet(tf.keras.Sequential):
    """
    Wrapper class for creating Keras network used for playing Atari games.
    """


    def __init__(self, input_shape, action_shape, net_conf, minval=-1, maxval=1):

        super().__init__()

        self.minval = minval
        self.maxval = maxval

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
        Returns weights shifted & scaled to the range [0,1] using the specified minimum & maximum weight value.
        """
        span = self.maxval-self.minval
        return (self.get_weights()-self.minval)/span


    def set_weights(self,weights):
        """
        Receives an array of np.arrays, converts to a list to set as weights for the agent.
        """
        super().set_weights(list(weights))


    def predict(self, observation, epsilon=0):
        """
        Returns action with highest activation in output layer.
        """
        activations = super().predict(observation.observation)
        if epsilon:
            if epsilon>np.random.rand():
                return np.random.randint(self.action_shape)
        return np.argmax(activations)


    def clip_weights(self):
        """
        Function clips weights outside the min-max value interval.
        """
        clipped_weights = []
        for layer in self.get_weights():
            layer = np.where(layer>self.maxval,self.maxval,layer)
            layer = np.where(layer<self.minval,self.minval,layer)
            clipped_weights.append(layer)
        self.set_weights(clipped_weights)