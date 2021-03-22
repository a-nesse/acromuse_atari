## Preprocessing for DQN
These files are from the TensorFlow Agents library and modified to comply with the method described by Mnih et al. (2015).

Also, the dtype used to store the frames from the environment is changed here to use np.float32 instead of np.uint8. The Keras network will not accept np.uint8 as input.
