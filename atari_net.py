import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers

class AtariNet(tf.keras.Sequential):
    """
    Wrapper class for creating Keras network used for playing Atari games.
    """


    def __init__(self, input_shape, action_shape, net_conf):

        super().__init__()

        initializer = initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
        
        self.action_shape = action_shape

        self.add(keras.Input(shape=input_shape))
        for c in net_conf['conv_layer_params']:
            self.add(layers.Conv2D(
                c[0],
                c[1],
                c[2],
                activation=net_conf['conv_activation'],
                kernel_initializer=initializer,
                bias_initializer=initializer))
        self.add(layers.Flatten())
        for fc in net_conf['fc_layer_params']:
            self.add(layers.Dense(
                fc, 
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


    def set_weights(self,weights):
        """
        Receives an array of np.arrays, converts to a list to set as weights for the agent.
        """
        super().set_weights(list(weights))


    def predict(self, observation, epsilon=0):
        activations = super().predict(observation.observation)
        if epsilon:
            if epsilon>np.random.rand():
                return np.random.randint(self.action_shape)
        return np.argmax(activations)

