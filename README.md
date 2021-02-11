# APPMAS
Master's Thesis in Applied Data Science 2021

The script 'AtariDQN' is the main script for training a DQN agent. 
The script 'AtariEvolution' will evolve N generations of agents of the same structure.
The net structure, DQN parameters and evo parameters are all held in the corresponding .config files.

The code does not work as there needs to be a small change in the TF agents library to one of the classes.
The preprocessing of the environment oututs the observations in an incompatible dtype. 