import json
import os
import pickle
import shutil
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
        self.elite = self.evo_conf['elite']

        self.save_name = self.evo_conf['save_name']

        self.py_env = suite_atari.load(environment_name=self.env_name)
        self.env = tf_py_environment.TFPyEnvironment(self.py_env)

        self.evo = AtariGen(self.evo_conf)

        self.agents = []
        self.net_shape = None

        self.scores = None

        self.spd = 0
        self.hpd = 0
        self.spd_avg = None
        self.hpd_avg = None
        self.weights = []
        self.hpd_contrib = None
        self.k1_pc = self.evo_conf['k1_pc']
        self.k2_pc = self.evo_conf['k2_pc']
        self.k_p_mut = self.evo_conf['k_p_mut']

        self.log = {}
        self.elite_agents = {}
        self.train_frames = 0


    def _save_model(self, agent, gen, nr):
        """
        Method for saving agent.
        """
        filepath = os.path.join(
            os.getcwd(),
            'saved_models_evo',
            self.save_name + '-' + str(gen) + '-' + str(nr))
        with open(filepath, 'wb') as f:
            pickle.dump(agent.get_weights(), f)


    def _load_model(self, gen, nr):
        """
        Method for loading agent.
        """
        filepath = os.path.join(
            os.getcwd(),
            'saved_models_evo',
            self.save_name + '-' + str(gen) + '-' + str(nr))
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
        time,
        max_score,
        exploration_size):
        """
        Method for logging data from training.
        """
        self.log[gen]=[
            time,
            self.train_frames,
            max_score,
            self.scores,
            self.spd,
            self.hpd,
            exploration_size]
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
            self.save_name + str(gen) + '-gen_params')
        with open(filepath, 'w') as f:
            json.dump(params, f)


    def _load_gen_measures(self,gen)
        """
        Method for loading generation measures and parameters used in training next generation.
        """
        filepath = os.path.join(
            os.getcwd(),
            'saved_models_evo',
            self.save_name + str(gen) + '-gen_params')
        with open(filepath, 'r') as f:
            params = json.load(f)
        return params


    def _save_gen(self,gen):
        """
        Saves a copy of the weights of the current generation of agents. 
        """
        for i, agt in enumerate(self.agents):
            self._save_model(agt,gen,i)


    def _load_gen(self, gen)
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
            self.scores,
            self.spd,
            self.hpd,
            p_c,
            p_mut_div,
            p_mut_fit,
            tour_size]
        self._write_gen_measures(gen,params)


    def load_checkpoint(self,gen):
        self._load_gen(gen)
        self.scores, self.spd, self.hpd, p_c, p_mut_div, p_mut_fit, tour_size = self._load_gen_measures(gen)
        return p_c, p_mut_div, p_mut_fit, tour_size


    def initialize_gen(self):
        obs_shape = tuple(self.env.observation_spec().shape)
        action_shape = self.env.action_spec().maximum - self.env.action_spec().minimum + 1
        self.agents = []
        for _ in range(self.n_agents):
            self.agents.append(AtariNet(obs_shape, action_shape, self.net_conf))
        #saving network shape
        self.net_shape = self.agents[0].get_weights().shape


    def _score_agent(self, agent, n_runs):
        score = 0.0
        frames = 0
        for _ in range(n_runs):
            score_run = 0.0
            obs = self.env.reset()
            while not obs.is_last():
                action = agent.predict(obs)
                obs = self.env.step(action)
                score_run += obs.reward.numpy()[0]
                frames += 1
            self.py_env.close()
            score += score_run
        return score/n_runs, frames


    def generate_scores(self,gen):
        max_score = 0.0
        tot_frames = 0
        for i, agt in enumerate(self.agents):
            print(i)
            score_i, frames_i = self._score_agent(agt, self.n_runs)
            tot_frames += frames_i
            self.scores[i] = score_i
            if score_i > max_score:
                max_score = score_i
        gen_elite_agents = np.argpartition(self.scores, -self.elite)[-self.elite:]
        print("Max score: {}".format(max_score))
        return tot_frames, max_score, gen_elite_agents


    def _find_avg_agent(self):
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


    def _calc_spd(self):
        "Method for calculation standard population diversity."
        gene_sum = np.zeros(self.net_shape)
        for agt in self.agents:
            gene_sum += (agt-self.spd_avg)**2
        std_gene = np.sqrt(gene_sum/len(self.agents))
        spd = np.sum(std_gene/self.spd_avg)/len(self.spd_avg)
        self.spd = spd


    def _calc_hpd(self):
        "Method for calculation healthy population diversity."
        self.hpd_contrib = np.zeros(len(self.agents))
        weighted_gene_sum = np.zeros(self.net_shape)
        for i, agt in enumerate(self.agents):
            sq_diff = (agt.get_weights()-self.hpd_avg)**2
            self.hpd_contrib[i] = self.weights[i]*np.sqrt(np.sum(sq_diff))
            weighted_gene_sum += self.weights[i]*sq_diff
        w_std_gene = np.sqrt(weighted_gene_sum)
        hpd = np.sum(w_std_gene/self.hpd_avg)/len(self.hpd_avg)
        self.hpd = hpd


    def _calc_pc(self,gen):
        "Calculates the probability of crossover given the SPD according to the ACROMUSE algorithm."
        return ((self.spd/0.4)*(self.k2_pc-self.k1_pc))+self.k1_pc


    def _calc_p_mut_fit(self):
        p_muts = []
        f_max = np.max(self.scores)
        f_min = np.min(self.scores)
        for score in self.scores:
            p_muts.append(self.k_p_mut*((f_max-score)/(f_max-f_min)))
        return np.array(p_muts)


    def calc_measures(self,gen):
        """
        Method that runs the calculations for the SPD and HPD measures.
        """
        self._find_avg_agent()
        self._calc_spd()
        self._calc_hpd()
        p_c = self._calc_pc(gen)
        p_mut_div = ((0.4-self.spd)/0.4)*self.k_p_mut
        p_mut_fit = self._calc_p_mut_fit()
        tour_size = (self.hpd/0.3)*self.n_agents
        return p_c, p_mut_div, p_mut_fit, tour_size


    def restart_training(self,gen):
        """
        Method for restarting training from saved agent checkpoint.
        """
        self.load_log_elite()
        time = self.log[gen][0]
        frames = self.log[gen][1]
        p_c, p_mut_div, p_mut_fit, tour_size = self.load_checkpoint(gen)
        return time, frames, p_c, p_mut_div, p_mut_fit, tour_size


    def evolve(self,restart_gen):
        """
        Method that develops agents through evolution.
        """
        start_time = time.time()
        self.initialize_gen()
        if restart_gen:
            print('Restarting from generation nr. {}'.format(restart_gen))
            restart_time, frames, p_c, p_mut_div, p_mut_fit, tour_size = self.restart_training()
            self.train_frames = frames
            start_time -= restart_time #adding time from previous training
        else:
            self.scores = np.zeros(self.n_agents)
            gen_frames, max_score, gen_elite_agents = self.generate_scores()
            self.elite_agents[gen] = gen_elite_agents
            self.train_frames += gen_frames
            self._calc_measures(gen)
            self.log_data()
            p_c, p_mut_div, p_mut_fit, tour_size = self._calc_measures(0)
        for i in range(self.n_gens-1-restart_gen):
            gen = i+1
            print('\nEvolving generation {} ...\n'.format(gen))
            new_agents, exploration_size = self.evo.new_gen(
                self.agents,
                self.scores,
                p_c,
                p_mut_div,
                p_mut_fit,
                tour_size,
                self.elite_agents)
            self.agents.clear()
            self.agents = new_agents
            print('Scoring ...')
            gen_frames, max_score, gen_elite_agents = self.generate_scores()
            self.elite_agents[gen] = gen_elite_agents
            p_c, p_mut_div, p_mut_fit, tour_size = self.calc_measures(gen)
            self.train_frames += gen_frames
            time = time.time() - start_time
            self.log_data(
                gen,
                time,
                max_score,
                exploration_size)
            self.checkpoint(
                gen,
                p_c,
                p_mut_div,
                p_mut_fit,
                tour_size)
        print('\nLast generation reached.\n')


def main(restart_gen,net_path=os.path.join('configs','net.config'),evo_path=os.path.join('configs','evo_preset.config')):
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