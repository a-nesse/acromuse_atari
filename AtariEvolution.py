import numpy as np
import gym, json, os
import pickle

#from tf_agents.environments import suite_atari
import suite_atari_mod as suite_atari
from tf_agents.environments import tf_py_environment

from AtariNet import AtariNet
from AtariGen import AtariGen

import tensorflow as tf

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
            except:
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

        self.save_k = self.evo_conf['save_k']
        self.save_name = self.evo_conf['save_name']

        self.conv_layer_params = [tuple(x) for x in self.net_conf['conv_layer_params']]
        self.fc_layer_params = tuple(self.net_conf['fc_layer_params'])

        self.py_env = suite_atari.load(environment_name=self.env_name)
        self.env = tf_py_environment.TFPyEnvironment(self.py_env)
        self.evo = AtariGen(self.evo_conf)

    def save_model(self, agent, gen, nr):
        """
        Method for saving agent.
        """
        filepath = os.path.join(os.getcwd(), 'saved_models', self.save_name + '-' + str(gen) + str(nr))
        with open(filepath, 'wb') as f:
            pickle.dump(agent, f)

    def load_model(self, name):
        """
        Method for loading agent.
        """
        filepath = os.path.join(os.getcwd(), 'saved_models', name)
        with open(filepath, 'rb') as f:
            agent = pickle.load(f)
        return agent

    def save_top(self,gen):
        """
        Method for saving top-k performing agents.
        """
        top_k = np.argpartition(self.probs, -self.save_k)[-self.save_k:]
        for i, idx in enumerate(top_k):
            self.save_model(self.agents[idx], gen, i)

    def _initialize_gen(self):
        obs_shape = tuple(self.env.observation_spec().shape)
        action_shape = self.env.action_spec().maximum - self.env.action_spec().minimum + 1
        self.agents = []
        for _ in range(self.n_agents):
            self.agents.append(AtariNet(obs_shape, action_shape, self.net_conf))

    def evolve(self):
        print('\nInitializing gen 1 ...\n')
        self._initialize_gen()
        self.probs = np.zeros(self.n_agents)
        self._generate_probs()
        for i in range(self.n_gens-1):
            print('\nEvolving generation {} ...\n'.format(i+1))
            new_agents = self.evo.new_gen(self.agents,self.probs)
            self.agents.clear()
            self.agents = new_agents
            print('Scoring ...')
            self._generate_probs()
            self.save_top(i)
        print('\nLast generation finished.\n')

    def _generate_probs(self):
        max_score = 0.0
        for i, a in enumerate(self.agents):
            s = self._score_agent(a, self.n_runs)
            max_score = s if s > max_score else max_score
            self.probs[i] = s
        print('Max score: {}'.format(max_score))
        self.probs = self.probs/np.sum(self.probs)

    def _score_agent(self, agent, n_runs):
        score = 0.0
        for _ in range(n_runs):
            score_run = 0.0
            obs = self.env.reset()
            i=0
            while not obs.is_last():
                i+=1
                print(i)
                self.env.render(mode='human')
                action = agent.predict(obs)
                obs = self.env.step(action)
                score_run += obs.reward.numpy()[0]
            print('\nend_while\n')
            self.py_env.close()
            score += score_run
        return score/n_runs

    def demo(self):
        best = np.argmax(self.probs)
        best_agent = self.agents[best]
        obs = self.env.reset()
        score = 0.0
        while not obs.is_last():
            self.env.render(mode='human')
            action = best_agent.predict(obs)
            obs = self.env.step(action)
            score += obs.reward
        self.py_env.close()
        print('The highest rated agent scored {} in this game.'.format(score))

def main():
    evolver = AtariEvolution('net.config','evo_preset.config')
    evolver.evolve()
    evolver.demo()


if __name__ == "__main__":
    main()
