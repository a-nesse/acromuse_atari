# ACROMUSE and DQN implementation for playing Atari games
The `atari_acromuse.py` script runs evolutionary training of agents using the ACROMUSE approach[1]. When restarting training, simply run `python atari_acromuse.py restart_generation` with `restart_generation` replaced by the generation number to restart training from. The agents are scored using a specified number of steps within the environment and the top n agents are evaluated to determine the elite. As default, the config is set to use 2000 steps to rank the agents and only evaluate the highest scoring as the elite agent. All agents are saved during training.

The `atari_demo.py` will run a demo of an agent within a specified environment. Use `python atari_demo.py environment_name agent_path optional_epsilon_value optional_net_config_path`. 

For the DQN implementation, check the `dqn_implementation` folder or run `python -m dqn_implementation.atari_dqn.py` from this parent directory.


### Configs
See the *configs* folder for configuration files for the models. Here training hyperparameters and network structure can be changed.


## Proposing a genetic algorithm for training agents to play Atari games
This code was written as part of a Master's Thesis in Applied Data Science in 2021. Below is the abstract from the thesis.

### Thesis abstract
This thesis attempts to implement a genetic algorithm for training agents within the Atari game environments. The training is performed on hardware of a widely available character, and so the results give an indication of how well these models perform on relatively inexpensive equipment available to many people. The Atari environment Space Invaders was chosen to train and test the models in. As a baseline, a Deep Q-Network (DQN) algorithm[2][3] is implemented within TensorFlow's TF-Agents framework[4]. The DQN is a popular model that has inspired many new algorithms and is often used as a comparison to alternative approaches. An adaptive genetic algorithm called ACROMUSE[1] was implemented and compared with the performance of the DQN within the environment. This algorithm adaptively determines crossover rates, mutation rates and tournament selection size. Using measures for diversity and fitness, two subpopulations are maintained to avoid converging toward local optima. Based on the results found here, the algorithm did not seem to converge or produce high-performing agents, and importantly performed worse than the DQN approach. The reasons for why this algorithm fails and why other genetic algorithms have succeeded are discussed. The large number of weight parameters present in the network seem to be a barrier to good performance. It is suggested that a parallel training approach is necessary to reach the number of agents and generations where a good solution could be found. It is also shown how the number of frames skipped in the environment had a significant impact on the performance of the baseline DQN model.


### References
[1] B. McGinley, J. Maher, C. O'Riordan, and F. Morgan,
"Maintaining Healthy Population Diversity Using Adaptive Crossover, Mutation, and Selection",
*IEEE Transactions on Evolutionary Computation* vol. 15, no. 5, pp. 692–714, 2011. doi:10.1109/TEVC.2010.2046173.\
[2]
V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller, 
"Playing atari with deep reinforcement learning,"
2013. arXiv: 1312.5602 [cs.LG].\
[3]
V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D.Wierstra, S. Legg, and D. Hassabis, 
"Human-level control through deep reinforcement learning," 
*Nature*, vol. 518, pp. 529-533, 2015. doi: https://doi.org/10.1038/nature14236.\
[4]
D. Hafner, J. Davidson, and V. Vanhoucke, 
"Tensorflow agents: Effcient batched reinforcement learning in tensorflow," 
2018. arXiv: 1709.02878[cs.LG].