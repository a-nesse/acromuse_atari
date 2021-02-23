import numpy as np
from copy import deepcopy

class AtariGen:
    """
    Class containing functions to evolve networks.
    """

    def __init__(self,evo_conf):
        self.evo_conf = evo_conf
        self.n_agents = self.evo_conf['n_agents']
        self.keep_elite = self.evo_conf['keep_elite']
        self.p_mut = self.evo_conf['p_mut']
        self.alg = 'self.' + self.evo_conf['alg']
        self.mut = 'self.' + self.evo_conf['mut']
        self.sd_mut = self.evo_conf['sd_mut']
        
        self.fin_alt = [False]
        if self.alg == 'self.alternating':
            self.alt1 = []
            self.alt2 = []
        
        if self.alg[:12] == 'self.k_point':
            self.k = self.evo_conf['k']
            self.k_idx = -1
            self.k_split = np.array([0])

    def _pick_agents(self,probs):
        return np.random.choice(self.n_agents, 2, replace=False, p=probs)

    def new_gen(self, agents, probs):
        """
        Function for creating new generation of agents.
        """
        new_agents = []
        # if keep_top >0
        if self.keep_elite:
            # keeps top k performing agents
            top_k = np.argpartition(probs, -self.keep_elite)[-self.keep_elite:]
            for idx in top_k:
                new_agents.append(deepcopy(agents[idx]))
        for _ in range((self.n_agents-self.keep_elite)//2):
            a1, a2 = self._pick_agents(probs)
            c1, c2 = self.create_children(agents[a1], agents[a2])
            new_agents.append(c1)
            new_agents.append(c2)
        return new_agents

    def create_children(self, agent1, agent2):
        nw_1 = []
        nw_2 = []
        if self.alg == 'self.k_point_1':
            self.k_split = np.sort(np.random.choice(np.arange(len(agent1.get_weights())), self.k, replace=False))
            print(self.k_split)
        for i in range(len(agent1.get_weights())):
            nlw_1 = eval(self.alg)(agent1.get_weights()[i], agent2.get_weights()[i], i)
            nlw_2 = eval(self.alg)(agent2.get_weights()[i], agent1.get_weights()[i], i)
            nw_1.append(eval(self.mut)(nlw_1))
            nw_2.append(eval(self.mut)(nlw_2))
        c_agent_1 = deepcopy(agent1)
        c_agent_2 = deepcopy(agent2)
        c_agent_1.set_weights(nw_1)
        c_agent_2.set_weights(nw_2)
        return c_agent_1, c_agent_2

    def uniform(self, arr1, arr2, idx):
        "Crossover algorithm selecting randomly from each parent."
        sel1 = np.random.randint(0,2,arr1.shape,dtype=bool)
        sel2 = ~sel1
        return np.zeros(arr1.shape) + (arr1*sel1) + (arr2*sel2)

    def alternating(self, arr1, arr2, idx):
        "Crossover algorithm selecting alternately from each parent."
        if self.fin_alt[idx]:
            return np.zeros(arr1.shape) + (arr1*self.alt1[idx]) + (arr2*self.alt2[idx])
        else:
            self.alt1.append(np.reshape(np.arange(np.prod(arr1.shape))%2==0, arr1.shape))
            self.alt2.append(~(self.alt1[-1]))
            self.fin_alt[-1] = True
            self.fin_alt.append(False)
            return np.zeros(arr1.shape) + (arr1*self.alt1[idx]) + (arr2*self.alt2[idx])

    def k_point_full(self, arr1, arr2, idx):
        """
        Crossover algorithm selecting from each parent with a split at k points.
        Splits weights between layers.
        """
        for i,s in enumerate(self.k_split):
            if idx < s:
                if i%2==0:
                    return arr1
                else:
                    return arr2
        if len(self.k_split)%2 == 0:
            return arr1
        else:
            return arr2

    def k_point_layer(self, arr1, arr2, idx):
        """
        Crossover algorithm selecting from each parent with a split at k points.
        Splits at highest level of each layer and selects new split for each layer.
        """
        if idx != self.k_idx:
            self.k_split = np.sort(np.random.choice(np.arange(arr1.shape[0]), self.k, replace=False))
        raise NotImplementedError

    def gaussian(self, arr):
        """
        Algorithm for applying normally distributed mutation to weights.
        Weights chosen with given mutation rate.
        """
        mut = np.random.random_sample(arr.shape)<self.p_mut
        mut_val = np.random.normal(0.0,self.sd_mut,arr.shape)
        return arr + (mut_val*mut)

def main():
    pass

if __name__ == "__main__":
    main()