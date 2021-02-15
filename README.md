# APPMAS
Master's Thesis in Applied Data Science 2021

The script 'AtariDQN' is the main script for training a DQN agent. The script 'AtariEvolution' will evolve N generations of agents of the same structure. The net structure, DQN parameters and evo parameters are all held in the corresponding .config files.

suite_atari_mod is a modified tf_agents script. It needed to be changed since the network (including the one used by tf_agents strangely) could not handle a dtype specified in one of the functions here.