## Preprocessing for OpenAI Gym's Atari Environments
These files are from the TensorFlow Agents library[1] and modified to comply with the method described by Mnih et.al in 2013[2] and 2015[3] where appropriate. The changes are listed at the top of each script.

Also, the dtype used to store the frames from the environment is changed here to use np.float32 instead of np.uint8. The Keras network will not accept np.uint8 as input.

Notice that there are two different preprocessing scripts, due to the differences in the training/experience collecting environment used by DQN and the evaluation environments used by both DQN and the ACROMUSE genetic algorithm.

### References
[1]
D. Hafner, J. Davidson, and V. Vanhoucke, 
"Tensorow agents: Effcient batched reinforcement learning in tensorow," 
2018. arXiv: 1709.02878[cs.LG].\
[2]<
V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller, 
"Playing atari with deep reinforcement learning,"
2013. arXiv: 1312.5602 [cs.LG].\
[3]
V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D.Wierstra, S. Legg, and D. Hassabis, 
"Human-level control through deep reinforcement learning," 
Nature, vol. 518, pp. 529-533, 2015. doi: https://doi.org/10.1038/nature14236.