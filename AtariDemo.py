import numpy as np
import gym
import time
import json
import os
import pickle
import time
import sys

import tensorflow as tf

from AtariNet import AtariNet
import suite_atari_mod as suite_atari 
from tf_agents.environments import tf_py_environment

class AtariDemo:
    """
    Class for demoing agents.
    """

    def __init__(self,env_name,conf_path):
        def _load_config(conf_path):
            try:
                assert os.path.exists(conf_path)
            except:
                print('The config file specified does not exist.')
            with open(conf_path, 'r') as f:
                conf = json.load(f)
            return conf

        self.net_conf = _load_config(conf_path)

        self.py_env = suite_atari.load(environment_name=env_name, eval_env=True)
        self.env = tf_py_environment.TFPyEnvironment(self.py_env)
 
        obs_shape = tuple(self.env.observation_spec().shape)
        action_shape = self.env.action_spec().maximum - self.env.action_spec().minimum + 1

        self.agent = AtariNet(obs_shape, action_shape, self.net_conf)

    def import_weights(self,agent_path):
        with open(agent_path, 'r') as f:
            weights = json.load(f)
        self.agent.set_weights(weights)

    def run(self):
        time_step = self.env.reset()
        score = 0.0
        while not time_step.is_last():
            action_step = self.agent.predict(time_step)
            time_step = self.env.step(action_step)
            score += time_step.reward
            self.env.render(mode='human')
            time.sleep(0.05)
        self.env.close()
        print('\nThe agent scored {:.2f}\n'.format(score[0]))

def main(agent_path,env_name,conf_path='net_config'):
    """
    Run demo of loaded agent.
    """
    demo = AtariDemo(env_name,conf_path)
    demo.import_weights(agent_path)
    demo.run()

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args)==3:
        main(args[1],args[0],args[2])
    else:
        main(args[1],args[0])