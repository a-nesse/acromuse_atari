import numpy as np
from copy import deepcopy

class AtariGen:
    """
    Class containing functions to evolve networks.
    """

    def __init__(self,evo_conf):
        self.evo_conf = evo_conf
        self.n_agents = self.evo_conf['n_agents']
        self.loc_p_mut = self.evo_conf['loc_p_mut']
        self.sd_mut = self.evo_conf['sd_mut']


    def _tournament(self,probs,n,size):
        """
        Runs tournament by randomly considering a given number of agents randomly chosen from the population, 
        and selecting the n agents with highest probability.
        """
        participants = np.random.choice(
            np.arange(len(probs)),
            size,
            replace=False)
        winners = np.argpartition(probs[participants], -n)[-n:]
        return participants[winners]


    def _uniform(self, arr1, arr2):
        """
        Crossover algorithm selecting randomly from each parent.
        """
        sel1 = np.random.randint(0,2,arr1.shape,dtype=bool)
        sel2 = ~sel1
        return np.zeros(arr1.shape) + (arr1*sel1) + (arr2*sel2)


    def _mutate(self,arr,p):
        """
        Algorithm for applying normally distributed mutation to weights.
        Weights chosen with given mutation rate.
        """
        mut = np.random.random_sample(arr.shape)<p
        mut_val = np.random.normal(0.0,self.sd_mut,arr.shape)
        return arr + (mut_val*mut)


    def _create_offspring(self,parent,n_layers,n,p_mut):
        nw = []
        if n == 2:
            for i in range(n_layers):
                nlw = self._uniform(parent[0].get_weights()[i], parent[1].get_weights()[i])
                nw.append(self._mutate(nlw,self.loc_p_mut))
            offspring = deepcopy(parent[0])
        else:
            for i in range(n_layers):
                nw.append(self._mutate(parent.get_weights()[i],p_mut))
            offspring = deepcopy(parent)
        offspring.set_weights(nw)
        return offspring


    def new_gen(self,agents,probs,pc,p_mut,tour_size,elite):
        '''
        Function for creating new generation of agents.
        '''
        new_agents = []
        n_layers = len(agents[0].get_weights())
        #carrying over elite agent
        new_agents.append(agents[elite])
        for _ in range(len(agents)-1):
            n = np.random.choice([1,2],1,p=[1-pc,pc]) #selecting whether to use crossover
            parent = self._tournament(probs,n,tour_size)
            offspring = self._create_offspring(parent,n_layers,n,p_mut)
            new_agents.append(offspring)
        return new_agents


def main():
    pass

if __name__ == "__main__":
    main()