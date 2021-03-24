import json
import os
import pickle
import numpy as np

from tf_agents.environments import tf_py_environment
from preprocessing import suite_atari_evo as suite_atari

from atari_net import AtariNet
from atari_gen import AtariGen


class AtariEvolution:
    """
    Class for evolving networks to play Atari games.
    """

    def __init__(self,
                 net_conf_path,
                 evo_conf_path
                 ):
        
        def _load_config(conf_path):
            try:
                assert os.path.exists(conf_path)
            except IOError:
                print('The config file specified does not exist.')
            with open(conf_path, 'r') as f:
                conf = json.load(f)
            return conf

        self.net_conf = _load_config(net_conf_path)
        self.evo_conf = _load_config(evo_conf_path)

        self.env_name = self.evo_conf['env_name']
        self.n_agents = self.evo_conf['n_agents']
        self.n_gens = self.evo_conf['n_gens']
        self.n_runs = self.evo_conf['n_runs']
        self.elite = self.evo_conf['elite']

        self.save_name = self.evo_conf['save_name']

        self.py_env = suite_atari.load(environment_name=self.env_name)
        self.env = tf_py_environment.TFPyEnvironment(self.py_env)

        self.evo = AtariGen(self.evo_conf)

        self.agents = []
        self.net_shape = None

        self.scores = None
        self.elite_agent = None

        self.spd = []
        self.hpd = []
        self.spd_avg = None
        self.hpd_avg = None
        self.weights = []
        self.hpd_contrib = None
        self.k1_pc = self.evo_conf['k1_pc']
        self.k2_pc = self.evo_conf['k2_pc']
        self.k_p_mut = self.evo_conf['k_p_mut']


    def save_model(self, agent, gen, nr):
        """
        Method for saving agent.
        """
        filepath = os.path.join(os.getcwd(), 'saved_models_evo', self.save_name + '-' + str(gen) + str(nr))
        with open(filepath, 'wb') as f:
            pickle.dump(agent, f)


    def load_model(self, name):
        """
        Method for loading agent.
        """
        filepath = os.path.join(os.getcwd(), 'saved_models_evo', name)
        with open(filepath, 'rb') as f:
            agent = pickle.load(f)
        return agent


    def save_elite(self,gen):
        """
        Method for saving top-k performing agents.
        """
        top_k = np.argpartition(self.scores, -self.elite)[-self.elite:]
        for i, idx in enumerate(top_k):
            self.save_model(self.agents[idx], gen, i)


    def _initialize_gen(self):
        obs_shape = tuple(self.env.observation_spec().shape)
        action_shape = self.env.action_spec().maximum - self.env.action_spec().minimum + 1
        self.agents = []
        for _ in range(self.n_agents):
            self.agents.append(AtariNet(obs_shape, action_shape, self.net_conf))
        #saving network shape
        self.net_shape = self.agents[0].get_weights().shape


    def _score_agent(self, agent, n_runs):
        score = 0.0
        for _ in range(n_runs):
            score_run = 0.0
            obs = self.env.reset()
            while not obs.is_last():
                action = agent.predict(obs)
                obs = self.env.step(action)
                score_run += obs.reward.numpy()[0]
            self.py_env.close()
            score += score_run
        return score/n_runs


    def _generate_scores(self):
        max_score = 0.0
        elite_agent = -1
        tot_score = 0.0
        for i, agt in enumerate(self.agents):
            print(i)
            score_i = self._score_agent(agt, self.n_runs)
            self.scores[i] = score_i
            if score_i > max_score:
                max_score = score_i
                elite_agent = i
            tot_score += score_i
        self.scores = self.scores/np.sum(self.scores)
        self.elite_agent = elite_agent
        avg_score = tot_score / self.n_agents
        print("Max score: {}   Avg score: {}".format(max_score,avg_score))


    def find_avg_agent(self):
        """
        Calculates average agents for SPD and HPD.
        Also saves HPD weights for the agents.
        """
        total_fit = np.sum(self.scores)
        spd_sum = np.zeros(self.net_shape)
        hpd_sum = np.zeros(self.net_shape)
        weights = []
        for i, agt in enumerate(self.agents):
            spd_sum += agt.get_weights()
            w_i = self.scores[i]/total_fit
            weights.append(w_i)
            hpd_sum += w_i*agt.get_weights()
        self.weights = weights
        self.spd_avg = spd_sum/len(self.agents)
        self.hpd_avg = hpd_sum


    def calc_spd(self):
        "Method for calculation standard population diversity."
        gene_sum = np.zeros(self.net_shape)
        for agt in self.agents:
            gene_sum += (agt-self.spd_avg)**2
        std_gene = np.sqrt(gene_sum/len(self.agents))
        spd = np.sum(std_gene/self.spd_avg)/len(self.spd_avg)
        self.spd.append(spd)


    def calc_hpd(self):
        "Method for calculation healthy population diversity."
        self.hpd_contrib = np.zeros(len(self.agents))
        weighted_gene_sum = np.zeros(self.net_shape)
        for i, agt in enumerate(self.agents):
            sq_diff = (agt.get_weights()-self.hpd_avg)**2
            self.hpd_contrib[i] = self.weights[i]*np.sqrt(np.sum(sq_diff))
            weighted_gene_sum += self.weights[i]*sq_diff
        w_std_gene = np.sqrt(weighted_gene_sum)
        hpd = np.sum(w_std_gene/self.hpd_avg)/len(self.hpd_avg)
        self.hpd.append(hpd)


    def _calc_pc(self,gen):
        "Calculates the probability of crossover given the SPD according to the ACROMUSE algorithm."
        return ((self.spd[gen]/0.4)*(self.k2_pc-self.k1_pc))+self.k1_pc


    def _calc_p_mut_fit(self):
        p_muts = []
        f_max = np.max(self.scores)
        f_min = np.min(self.scores)
        for score in self.scores:
            p_muts.append(self.k_p_mut*((f_max-score)/(f_max-f_min)))
        return np.array(p_muts)


    def _calc_measures(self,gen):
        "Method that runs the calculations for the SPD and HPD measures."
        self.find_avg_agent()
        self.calc_spd()
        self.calc_hpd()
        p_c = self._calc_pc(gen)
        p_mut_div = ((0.4-self.spd[-1])/0.4)*self.k_p_mut
        p_mut_fit = self._calc_p_mut_fit()
        tour_size = (self.hpd[gen]/0.3)*self.n_agents
        return p_c, p_mut_div, p_mut_fit, tour_size


    def evolve(self):
        """
        Method that develops agents through evolution.
        """
        self._initialize_gen()
        self.scores = np.zeros(self.n_agents)
        self._generate_scores()
        p_c, p_mut_div, p_mut_fit, tour_size = self._calc_measures(0)
        for i in range(self.n_gens-1):
            gen = i+1
            print('\nEvolving generation {} ...\n'.format(gen))
            new_agents = self.evo.new_gen(self.agents,self.scores,p_c,p_mut_div,p_mut_fit,tour_size,self.elite_agent)
            self.agents.clear()
            self.agents = new_agents
            print('Scoring ...')
            self._generate_scores()
            self._calc_measures(gen)
            self.save_elite(gen)
        print('\nLast generation finished.\n')


def main():
    evolver = AtariEvolution(os.path.join('configs','net.config'),os.path.join('configs','evo_preset.config'))
    evolver.evolve()


if __name__ == "__main__":
    main()
