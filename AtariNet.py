import numpy as np
import gym, time, json, os

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers

class AtariNet(tf.keras.Sequential):
    """
    Wrapper class for creating Keras network used for playing Atari games.
    """

    def __init__(self, input_shape, action_shape, net_conf):

        super(AtariNet, self).__init__()

        self.add(tf.keras.Input(shape=input_shape))
        for c in net_conf['conv_layer_params']:
            self.add(tf.keras.layers.Conv2D(c[0],c[1],c[2],activation=net_conf['conv_activation'], kernel_initializer=eval(net_conf['initializer'])))
        self.add(tf.keras.layers.Flatten())
        for fc in net_conf['fc_layer_params']:
            self.add(tf.keras.layers.Dense(fc, activation=net_conf['fc_activation'], kernel_initializer=eval(net_conf['initializer'])))
        self.add(tf.keras.layers.Dense(action_shape, activation=net_conf['action_activation'], kernel_initializer=eval(net_conf['initializer'])))
        self.build()

    def predict(self,observation):
        activations = super().predict(observation.observation)
        return np.argmax(activations)