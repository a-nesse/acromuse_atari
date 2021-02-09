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
        for i in range(len(agent1.get_weights())):
            nlw_1 = self.uniform(agent1.get_weights()[i], agent2.get_weights()[i])
            nlw_2 = self.uniform(agent2.get_weights()[i], agent1.get_weights()[i])
            nw_1.append(self.mutate(nlw_1,self.p_mut))
            nw_2.append(self.mutate(nlw_2,self.p_mut))
        c_agent_1 = deepcopy(agent1)
        c_agent_2 = deepcopy(agent2)
        c_agent_1.set_weights(nw_1)
        c_agent_2.set_weights(nw_2)
        return c_agent_1, c_agent_2

    def uniform(self, arr1, arr2):
        sel1 = np.random.randint(0,2,arr1.shape,dtype=bool)
        sel2 = ~sel1
        return np.zeros(arr1.shape) + (arr1*sel1) + (arr2*sel2)

    def mutate(self, arr, p_mut):
        mut = np.random.random_sample(arr.shape)<p_mut
        mut_val = np.random.normal(0.0,1.0,arr.shape)
        return arr + (mut_val*mut)

def main():
    pass

if __name__ == "__main__":
    main()