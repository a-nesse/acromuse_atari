import numpy as np
from copy import deepcopy

class AtariGen:
    """
    Class containing functions to evolve networks.
    """

    def __init__(self,evo_conf):
        self.evo_conf = evo_conf
        self.n_agents = self.evo_conf['n_agents']
        self.p_mut_loc = self.evo_conf['p_mut_loc']
        self.sd_mut = self.evo_conf['sd_mut']
        self.k_p_mut = self.evo_conf['k_p_mut']


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


    def _create_offspring(self,agents,parent,n_layers,p_mut):
        """
        Function to create offpsring.
        """
        n_w = []
        n_parent = len(parent)
        if n_parent == 2:
            for i in range(n_layers):
                nlw = self._uniform(agents[parent[0]].get_weights()[i], agents[parent[1]].get_weights()[i])
                n_w.append(self._mutate(nlw,self.p_mut_loc))
            offspring = deepcopy(parent[0])
        else:
            for i in range(n_layers):
                n_w.append(self._mutate(agents[parent].get_weights()[i],p_mut))
            offspring = deepcopy(parent)
        offspring.set_weights(n_w)
        return offspring


    def _calc_p_mut(self,parent,p_mut_div,p_mut_fit):
        """
        Calculate the mutation rate.
        """
        if len(parent)==2:
            return 0
        else:
            return (p_mut_fit[parent]+p_mut_div)/2


    def new_gen(self,agents,probs,p_c,p_mut_div,p_mut_fit,tour_size,elite):
        '''
        Function for creating new generation of agents.
        '''
        new_agents = []
        n_layers = len(agents[0].get_weights())
        #carrying over elite agent(s)
        for agt in elite:
            new_agents.append(agents[agt])
        exploration_size = 0
        for _ in range(len(agents)-1):
            n_parent = np.random.choice([1,2],1,p=[1-p_c,p_c]) #selecting whether to use crossover
            exploration_size += 2-n_parent #counting members of exploration population
            parent = self._tournament(probs,n_parent,tour_size)
            p_mut = self._calc_p_mut(parent,p_mut_div,p_mut_fit)
            offspring = self._create_offspring(agents,parent,n_layers,p_mut)
            new_agents.append(offspring)
        return new_agents, exploration_size


def main():
    pass

if __name__ == "__main__":
    main()