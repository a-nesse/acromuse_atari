# DQN Implementation

To run this DQN implementation, navigate to the parent directory and run the command `python -m dqn_implementation.atari_dqn.py`.

This is an attempt at implementing the deep-Q network approach to playing Atari games, described in the paper "Playing Atari with Deep Reinforcement Learning" (Mnih et.al 2013).

The implementation was made with the purpose of comparing the developed genetic algorithm within the TensorFlow framework. The implementation employs the TensorFlow Agents library, which contains a deep Q agent, step driver and replay buffer used in the DQN class.

## Issues

This implementation is not the focus of the project, but just used as a comparison since it is an established method and used as a baseline in many papers. 
The model has not been tested on any other game than SpaceInvaders and is not guaranteed to match the results from the paper (Mnih et.al 2013) on the other environments.

### Replay Buffer Storage

There is not an option to store frames from the game individually using the TF Agents Uniform Replay Buffer. This means that each of the 4 frame stacks used for training the networks, have to be stored as a stack, instead of extracted from the buffer individually and placed into a stack. Because of this, every frame in the replay buffer is stored 4 times. This inefficient use of memory caused it to not be possible for me to use a 1 mill frame replay buffer with the 32GB of memory that was available in the machine used for training.
This is a known issue and they do provide the PyHashedReplayBuffer class, which will store the frames individually, but it was not possible to simply use this buffer with the DQN agent class. There might well be a way to combine these, but making this work was not within the scope of time in this project.
A replay buffer of size 100 000 frames is used instead, which corresponds to almost 2 hours worth of real-time gameplay.