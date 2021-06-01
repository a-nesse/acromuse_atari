# DQN Implementation

To run this DQN implementation, navigate to the parent directory and run the command `python -m dqn_implementation.atari_dqn.py`. When restarting training, simply run `python -m dqn_implementation.atari_dqn.py restart_step` from teh parent directory with `restart_step` replaced by the step number to restart training from. 

This is an attempt at implementing a somewhat modified deep-Q network approach to playing Atari games, described in the papers "Playing Atari with Deep Reinforcement Learning"[1] and "Human-level control through deep reinforcement learning"[2]. See the config files to for the network structure and hyperparameters used, which deviate somewhat from the papers.

The implementation was made with the purpose of comparing the developed genetic algorithm within the TensorFlow framework. The implementation employs the TensorFlow Agents library, which contains a deep Q agent, step driver and replay buffer used in the DQN class.

### Disclaimer

This implementation is not the focus of the project, but just used as a comparison since it is an established method and used as a baseline in many papers. 
The model has not been tested on any other game than SpaceInvaders and is not guaranteed to match the results from the 2013 paper [1] on the other environments.

### References
[1]
V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller, 
"Playing atari with deep reinforcement learning,"
2013. arXiv: 1312.5602 [cs.LG].\
[2]
V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D.Wierstra, S. Legg, and D. Hassabis, 
"Human-level control through deep reinforcement learning," 
*Nature*, vol. 518, pp. 529-533, 2015. doi: https://doi.org/10.1038/nature14236.