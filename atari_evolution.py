import json
import math
import os
import pickle
import sys
import time
import numpy as np

from tf_agents.environments import tf_py_environment
from preprocessing import suite_atari_evo as suite_atari

from atari_net import AtariNet
from atari_gen import AtariGen


class AtariEvolution:
    """
    Class for evolving networks to play Atari games.
    """

    def __init__(
        self,
        net_conf_path,
        evo_conf_path):
        
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

        self.save_name = self.evo_conf['save_name']

        # environment for ranking agents & evaluating elite
        self.py_env = suite_atari.load(environment_name=self.env_name, eval_env=True)
        self.env = tf_py_environment.TFPyEnvironment(self.py_env)

        self.obs_shape = tuple(self.env.observation_spec().shape)
        self.action_shape = self.env.action_spec().maximum - self.env.action_spec().minimum + 1

        self.evo = AtariGen(self.evo_conf,self.net_conf,self.obs_shape,self.action_shape)

        self.agents = []
        self.net_shape = None

        self.scores = np.zeros(self.n_agents)
        self.highest_score = [[0,0,0],[0,0,0]]

        self.spd = 0.0
        self.hpd = 0.0
        self.spd_avg = None
        self.hpd_avg = None
        self.weights = []
        self.hpd_contrib = None
        self.k1_pc = self.evo_conf['k1_pc']
        self.k2_pc = self.evo_conf['k2_pc']
        self.k_p_mut = self.evo_conf['k_p_mut']
        self.adaptive_measures = bool(self.evo_conf['adaptive_measures'])
        self.spd_max = self.evo_conf['spd_max'] 
        self.hpd_max = self.evo_conf['hpd_max']
        self.t_size_max = self.n_agents/6

        self.log = {}
        self.elite_agents = {}
        self.train_steps = 0
        self.n_weights = 0

        self.n_eval_runs = self.evo_conf['n_eval_runs']
        self.epsilon = self.evo_conf['epsilon']


    def _save_model(self, agent, gen, num):
        """
        Method for saving agent.
        """
        filepath = os.path.join(
            os.getcwd(),
            'saved_models_evo',
            self.save_name + '-' + str(gen) + '-' + str(num))
        with open(filepath, 'wb') as f:
            pickle.dump(agent.get_weights(), f)


    def _load_model(self, gen, num):
        """
        Method for loading agent.
        """
        filepath = os.path.join(
            os.getcwd(),
            'saved_models_evo',
            self.save_name + '-' + str(gen) + '-' + str(num))
        with open(filepath, 'rb') as f:
            agent = pickle.load(f)
        return agent


    def _write_log(self):
        """
        Method for writing log to file.
        """
        filepath = os.path.join(
            os.getcwd(),
            'saved_models_evo',
            self.save_name + '-log')
        with open(filepath, 'w') as f:
            json.dump(self.log, f)


    def _write_elite_dict(self):
        """
        Method for saving top-k performing agents.
        """
        filepath = os.path.join(
            os.getcwd(),
            'saved_models_evo',
            self.save_name + '-elites')
        with open(filepath, 'w') as f:
            json.dump(self.elite_agents, f)


    def log_data(
        self,
        gen,
        gen_time,
        elite_avg,
        elite_max,
        exploration_size):
        """
        Method for logging data from training.
        """
        # logging highest performing elite agent across generations
        if elite_avg>self.highest_score[0][2]:
            self.highest_score[0] = [gen,self.elite_agents[str(gen)],elite_avg]
        if elite_max>self.highest_score[1][2]:
            self.highest_score[1] = [gen,self.elite_agents[str(gen)],elite_max]
        self.log[str(gen)]=[
            gen_time,
            self.train_steps,
            elite_avg,
            elite_max,
            self.spd,
            self.hpd,
            exploration_size,
            self.highest_score]
        self._write_log()
        self._write_elite_dict()


    def load_log_elite(self):
        """
        Method for loading log & list of elites from file.
        """
        filepath = os.path.join(
            os.getcwd(),
            'saved_models_evo',
            self.save_name + '-log')
        with open(filepath, 'r') as f:
            self.log = json.load(f)
        filepath = os.path.join(
            os.getcwd(),
            'saved_models_evo',
            self.save_name + '-elites')
        with open(filepath, 'r') as f:
            self.elite_agents = json.load(f)


    def _write_gen_measures(self,gen,params):
        """
        Method for writing generation measures and parameters used in training next generation.
        """
        filepath = os.path.join(
            os.getcwd(),
            'saved_models_evo',
            self.save_name + '-' + str(gen) + '-gen_params')
        with open(filepath, 'w') as f:
            json.dump(params, f)


    def _load_gen_measures(self,gen):
        """
        Method for loading generation measures and parameters used in training next generation.
        """
        filepath = os.path.join(
            os.getcwd(),
            'saved_models_evo',
            self.save_name + '-' + str(gen) + '-gen_params')
        with open(filepath, 'r') as f:
            params = json.load(f)
        return params


    def _save_gen(self,gen):
        """
        Saves a copy of the weights of the current generation of agents. 
        """
        for i, agt in enumerate(self.agents):
            self._save_model(agt,gen,i)


    def _load_gen(self, gen):
        """
        Loads the saved weights of the specified generation. 
        """
        for i in range(self.n_agents):
            n_w = self._load_model(gen, i)
            self.agents[i].set_weights(n_w)


    def checkpoint(
        self,
        gen,
        p_c,
        p_mut_div,
        p_mut_fit,
        tour_size):
        """
        Method for saving current generation for restarting, in case of training stop.
        """
        self._save_gen(gen)
        params = [
            list(self.scores),
            self.spd,
            self.hpd,
            p_c,
            p_mut_div,
            p_mut_fit,
            tour_size]
        self._write_gen_measures(gen,params)


    def load_checkpoint(self,gen):
        """
        Loads specified saved generation of agents and measures of that generation.
        """
        self._load_gen(gen)
        scores, self.spd, self.hpd, p_c, p_mut_div, p_mut_fit, tour_size = self._load_gen_measures(gen)
        self.scores = np.array(scores)
        return p_c, p_mut_div, p_mut_fit, tour_size


    def zero_net(self):
        """
        Returns an array containing 0-arrays with same shape as net weights.
        """
        zero_net = []
        for layer in self.net_shape:
            zero_net.append(np.zeros(layer))
        return np.array(zero_net,dtype=object)


    def _save_net_shape(self):
        "Method for saving shape of network."
        shapes = []
        for w in self.agents[0].get_weights():
            shapes.append(w.shape)
        self.net_shape = shapes


    def _calc_n_weights(self):
        "Method for saving number of weights/biases in network"
        n_weights = 0
        for layer in self.net_shape:
            l_weights = 1
            for dim in layer:
                l_weights *= dim
            n_weights += l_weights
        self.n_weights = n_weights


    def _score_agent(self, agent, n_runs, eval_elite=False):
        """
        Score one agent on the environment.
        Returns the median score.
        """
        steps = 0
        scores = np.zeros(n_runs)
        max_ep_score = 0.0
        for i in range(n_runs):
            obs = self.env.reset()
            while not obs.is_last():
                action = agent.action(obs,epsilon=self.epsilon)
                obs = self.env.step(action)
                scores[i] += obs.reward.numpy()[0]
                steps += 1
        max_ep_score = np.max(scores)
        if eval_elite:
            # using average score for evaluation of elite
            agt_score = np.average(scores)
            print('\nEvaluation step count: {}\n'.format(steps))
        else:
            # use median score for ranking agents
            agt_score = np.median(scores)
        return float(agt_score), float(max_ep_score), int(steps)


    def generate_scores(self):
        """
        Generate scores for all agents in the generation.
        """
        tot_steps= 0
        for i, agt in enumerate(self.agents):
            print('Scoring agent {}...  '.format(i+1))
            score_i, _, steps_i = self._score_agent(agt, self.n_runs)
            tot_steps += steps_i
            self.scores[i] = score_i
            print(score_i)
        gen_elite_agent = int(np.argmax(self.scores))
        return tot_steps, gen_elite_agent


    def eval_score(self,elite_agent):
        """
        Function to run evaluation of elite agent.
        """
        avg_score, max_score, _ = self._score_agent(self.agents[elite_agent],self.n_eval_runs,eval_elite=True)
        print('Elite agent evaluation:\nAverage score: {}\nMax episode score: {}\n'.format(avg_score,max_score))
        return avg_score, max_score


    def _arr_sum(self,arr):
        "Function summing all elements in array of np.arrays"
        tot_sum = 0
        for a in arr:
            tot_sum += np.sum(a)
        return tot_sum
    

    def _arr_sqrt(self,arr):
        "Function finding the square root of all elements in array of np.arrays"
        nw = []
        for a in arr:
            nw.append(np.sqrt(a))
        return np.array(nw,dtype=object)


    def _find_avg_agent(self):
        """
        Calculates average agents for SPD and HPD.
        Also saves HPD weights for the agents.
        """
        total_fit = np.sum(self.scores)
        spd_sum = self.zero_net()
        hpd_sum = self.zero_net()
        weights = []
        for i, agt in enumerate(self.agents):
            spd_sum += agt.get_scaled_weights()
            w_i = self.scores[i]/total_fit
            weights.append(w_i)
            hpd_sum += w_i*agt.get_scaled_weights()
        self.weights = weights
        self.spd_avg = spd_sum/len(self.agents)
        self.hpd_avg = hpd_sum


    def _calc_spd(self):
        "Method for calculation standard population diversity."
        gene_sum = self.zero_net()
        for agt in self.agents:
            gene_sum += (agt.get_scaled_weights()-self.spd_avg)**2
        std_gene = self._arr_sqrt(gene_sum/self.n_agents)
        spd = self._arr_sum(std_gene/self.spd_avg)/self.n_weights
        self.spd = spd
        if spd > self.spd_max and self.adaptive_measures:
            # setting the max to the highest seen SPD vlaue
            self.spd_max = spd


    def _calc_hpd(self):
        "Method for calculation healthy population diversity."
        self.hpd_contrib = np.zeros(self.n_agents)
        weighted_gene_sum = self.zero_net()
        for i, agt in enumerate(self.agents):
            sq_diff = (agt.get_scaled_weights()-self.hpd_avg)**2
            self.hpd_contrib[i] = self.weights[i]*np.sqrt(self._arr_sum(sq_diff))
            weighted_gene_sum += self.weights[i]*sq_diff
        w_std_gene = self._arr_sqrt(weighted_gene_sum)
        hpd = self._arr_sum(w_std_gene/self.hpd_avg)/self.n_weights
        self.hpd = hpd
        if hpd > self.hpd_max and self.adaptive_measures:
            # setting the max to the highest seen HPD vlaue
            self.hpd_max = hpd


    def _calc_pc(self):
        "Calculates the probability of crossover given the SPD according to the ACROMUSE algorithm."
        spd_lim = self.spd_max if self.spd>self.spd_max else self.spd
        return ((spd_lim/0.4)*(self.k2_pc-self.k1_pc))+self.k1_pc


    def _calc_p_mut_fit(self):
        p_muts = []
        f_max = np.max(self.scores)
        f_min = np.min(self.scores)
        for score in self.scores:
            p_muts.append(self.k_p_mut*((f_max-score)/(f_max-f_min)))
        return p_muts


    def calc_measures(self):
        """
        Method that runs the calculations for the SPD and HPD measures.
        """
        self._find_avg_agent()
        self._calc_spd()
        self._calc_hpd()
        p_c = self._calc_pc()
        spd_lim = self.spd_max if self.spd>self.spd_max else self.spd
        hpd_lim = self.hpd_max if self.hpd>self.hpd_max else self.hpd
        p_mut_div = ((0.4-spd_lim)/0.4)*self.k_p_mut
        p_mut_fit = self._calc_p_mut_fit()
        tour_size = math.ceil((hpd_lim/0.3)*self.t_size_max)
        return p_c, p_mut_div, p_mut_fit, tour_size


    def restart_training(self,gen):
        """
        Method for restarting training from saved agent checkpoint.
        """
        self.load_log_elite()
        gen_time = self.log[str(gen)][0]
        steps = self.log[str(gen)][1]
        p_c, p_mut_div, p_mut_fit, tour_size = self.load_checkpoint(gen)
        return gen_time, steps, p_c, p_mut_div, p_mut_fit, tour_size


    def initialize_gen(self,start_time,restart_gen):
        """
        Method for initializing first generation of agents.
        """
        self.agents = []
        for _ in range(self.n_agents):
            self.agents.append(AtariNet(self.obs_shape, self.action_shape, self.net_conf))
        #saving network shape & number of genes
        self._save_net_shape()
        self._calc_n_weights()
        if not restart_gen:
            #for initializing generation zero
            gen_steps, gen_elite_agent = self.generate_scores()
            self.elite_agents[str(0)] = gen_elite_agent
            p_c, p_mut_div, p_mut_fit, tour_size = self.calc_measures()
            self.train_steps += gen_steps
            elite_avg, elite_max = self.eval_score(gen_elite_agent)
            gen_time = time.time() - start_time
            self.log_data(
                0,
                gen_time,
                elite_avg,
                elite_max,
                self.n_agents)
            self.checkpoint(
                0,
                p_c,
                p_mut_div,
                p_mut_fit,
                tour_size)
            print('\nSPD: {}\nHPD: {}\n'.format(self.spd,self.hpd))
            return gen_time, gen_steps, p_c, p_mut_div, p_mut_fit, tour_size
        else:
            print('Restarting from generation nr. {}'.format(restart_gen))
            restart_time, restart_steps, p_c, p_mut_div, p_mut_fit, tour_size = self.restart_training(restart_gen)
            self.train_steps = restart_steps
            return restart_time, 0, p_c, p_mut_div, p_mut_fit, tour_size


    def evolve(self,restart_gen):
        """
        Method that develops agents through evolution.
        """
        start_time = time.time()
        print('Initializing...')
        gen_time, _, p_c, p_mut_div, p_mut_fit, tour_size = self.initialize_gen(start_time,restart_gen)
        if restart_gen:
            start_time -= gen_time
        for i in range(self.n_gens-1-restart_gen):
            gen = i+1+restart_gen
            print('\nEvolving generation {} ...\n'.format(gen))
            new_agents, exploration_size = self.evo.new_gen(
                self.agents,
                self.scores,
                p_c,
                p_mut_div,
                p_mut_fit,
                tour_size,
                self.elite_agents[str(gen-1)])
            self.agents.clear()
            self.agents = new_agents
            print('Scoring ...')
            gen_steps, gen_elite_agent = self.generate_scores()
            self.elite_agents[str(gen)] = gen_elite_agent
            p_c, p_mut_div, p_mut_fit, tour_size = self.calc_measures()
            self.train_steps += gen_steps
            elite_avg, elite_max = self.eval_score(gen_elite_agent)
            gen_time = time.time() - start_time
            self.log_data(
                gen,
                gen_time,
                elite_avg,
                elite_max,
                exploration_size)
            self.checkpoint(
                gen,
                p_c,
                p_mut_div,
                p_mut_fit,
                tour_size)
            print('\nSPD: {}\nHPD: {}\n'.format(self.spd,self.hpd))
        print('\nLast generation reached.\n')


def main(restart_gen):
    """
    Main function runs evolution using the config files.
    """
    net_path=os.path.abspath(os.path.join('configs','net.config'))
    evo_path=os.path.abspath(os.path.join('configs','evo_preset.config'))
    evolver = AtariEvolution(net_path,evo_path)
    if not os.path.isdir(os.path.join(os.getcwd(),'saved_models_evo')):
        os.makedirs(os.path.join(os.getcwd(),'saved_models_evo'))
    evolver.evolve(restart_gen)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args)==1:
        main(int(args[0]))
    else:
        main(0)