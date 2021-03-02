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

        initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='untruncated_normal')
        
        self.action_shape = action_shape

        self.add(tf.keras.Input(shape=input_shape))
        for c in net_conf['conv_layer_params']:
            self.add(tf.keras.layers.Conv2D(c[0],c[1],c[2],activation=net_conf['conv_activation'], kernel_initializer=initializer))
        self.add(tf.keras.layers.Flatten())
        for fc in net_conf['fc_layer_params']:
            self.add(tf.keras.layers.Dense(fc, activation=net_conf['fc_activation'], kernel_initializer=initializer))
        self.add(tf.keras.layers.Dense(action_shape, activation=None, kernel_initializer=initializer))
        self.build()


    def predict(self,observation, epsilon=0.05):
        activations = super().predict(observation.observation)
        if epsilon:
            return np.random.randint(self.action_shape)
        return np.argmax(activations)

