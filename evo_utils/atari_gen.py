import numpy as np
from evo_utils.atari_net import AtariNet

class AtariGen:
    """
    Class containing functions for creating a new generation of agents, 
    given a set of agents and measures.
    """

    def __init__(self,evo_conf,net_conf,obs_shape,action_shape):
        """
        Initializes a AtariGen object.
        """
        self.evo_conf = evo_conf
        self.net_conf = net_conf
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.n_agents = self.evo_conf['n_agents']
        self.p_mut_loc = self.evo_conf['p_mut_loc']
        self.k_p_mut = self.evo_conf['k_p_mut']
        self.minval = self.evo_conf['net_minval']
        self.maxval = self.evo_conf['net_maxval']
        self.val_buffer = self.evo_conf['val_buffer']


    def _tournament(self,probs,n,size):
        """
        Runs tournament by randomly considering a given number of agents randomly chosen from the population,
        and selecting the n agents with highest probability.
        """
        participants = np.random.choice(
            self.n_agents,
            size=size,
            replace=False)
        winners = np.argpartition(probs[participants], -n)[-n:]
        return participants[winners]


    def _uniform(self, arr1, arr2):
        """
        Crossover algorithm selecting randomly from each parent.
        """
        sel1 = np.random.randint(0,2,arr1.shape,dtype=bool)
        sel2 = ~sel1
        return (arr1*sel1) + (arr2*sel2)


    def _mutate(self,arr,p_mut):
        """
        Algorithm for applying normally distributed mutation to weights.
        Weights chosen with given mutation rate.
        """
        mut = np.random.random_sample(arr.shape)<p_mut
        no_mut = ~mut
        mut_val = np.random.uniform(low=self.minval,high=self.maxval,size=arr.shape)
        return (no_mut*arr) + (mut*mut_val)


    def _create_offspring(self,agents,parent,n_layers,p_mut):
        """
        Function to create offpsring.
        """
        n_w = []
        n_parent = len(parent)
        if n_parent == 2:
            for i in range(n_layers):
                nlw = self._uniform(agents[parent[0]].get_weights()[i], agents[parent[1]].get_weights()[i])
                nlw = self._mutate(nlw,p_mut)
                n_w.append(nlw)
        else:
            for i in range(n_layers):
                nlw = self._mutate(agents[parent[0]].get_weights()[i],p_mut)
                n_w.append(nlw)
        offspring = AtariNet(
            self.obs_shape,
            self.action_shape,
            self.net_conf,
            val_buffer=self.val_buffer)
        offspring.set_weights(n_w)
        return offspring


    def _calc_p_mut(self,parent,p_mut_div,p_mut_fit):
        """
        Calculate the mutation rate.
        """
        if len(parent)==2:
            return self.p_mut_loc
        else:
            return (p_mut_fit[int(parent[0])]+p_mut_div)/2


    def new_gen(self,agents,probs,p_c,p_mut_div,p_mut_fit,tour_size,elite):
        """
        Function for creating new generation of agents.
        """
        new_agents = []
        n_layers = len(agents[0].get_weights())
        # carrying over elite agent
        new_agents.append(AtariNet(
            self.obs_shape,
            self.action_shape,
            self.net_conf))
        new_agents[-1].set_weights(agents[elite].get_weights())
        exploration_size = 0
        for _ in range(len(agents)-1):
            n_parent = np.random.choice([1,2],1,p=[1-p_c,p_c])[0] # selecting whether to use crossover
            exploration_size += int(2-n_parent) # counting members of exploration population
            parent = self._tournament(probs,n_parent,tour_size)
            p_mut = self._calc_p_mut(parent,p_mut_div,p_mut_fit)
            offspring = self._create_offspring(agents,parent,n_layers,p_mut)
            new_agents.append(offspring)
        return new_agents, exploration_size
