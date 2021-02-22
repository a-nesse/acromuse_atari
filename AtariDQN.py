import numpy as np
import gym
import time
import json
import os
import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
#from tf_agents.environments import suite_atari
import suite_atari_mod as suite_atari 
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

class AtariDQN:
    """
    Class for training Deep-Q agent to play Atari games.
    """

    def __init__(self, net_conf_path='',dqn_conf_path=''):
        
        def _load_config(conf_path):
            try:
                assert os.path.exists(conf_path)
            except:
                print('The config file specified does not exist.')
            with open(conf_path, 'r') as f:
                conf = json.load(f)
            return conf

        self.net_conf = _load_config(net_conf_path)
        self.dqn_conf = _load_config(dqn_conf_path)

        self.num_iterations = self.dqn_conf['num_iterations']
        self.initial_collect_steps = self.dqn_conf['initial_collect_steps']
        self.collect_steps_per_iteration = self.dqn_conf['collect_steps_per_iteration']
        self.replay_buffer_max_length = self.dqn_conf['replay_buffer_max_length'] 
        self.batch_size = self.dqn_conf['batch_size']
        self.learning_rate = self.dqn_conf['learning_rate']
        self.log_interval = self.dqn_conf['log_interval']
        self.num_eval_episodes = self.dqn_conf['num_eval_episodes']
        self.eval_interval = self.dqn_conf['eval_interval']

        self.train_py_env = suite_atari.load(environment_name=self.dqn_conf['env_name'])
        self.eval_py_env = suite_atari.load(environment_name=self.dqn_conf['env_name'])
        self.train_env = tf_py_environment.TFPyEnvironment(self.train_py_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(self.eval_py_env)

        self.obs_spec = self.train_env.observation_spec()
        self.action_spec = self.train_env.action_spec()
        self.step_spec = self.train_env.time_step_spec()

        self.q_net = q_network.QNetwork(self.obs_spec,self.action_spec,conv_layer_params=[tuple(c) for c in self.net_conf['conv_layer_params']],fc_layer_params=tuple(self.net_conf['fc_layer_params']))
        self.optimizer = eval(self.dqn_conf['optimizer'])(learning_rate=self.dqn_conf['learning_rate'])
        self.train_step_counter = tf.Variable(0)

        self.agent = dqn_agent.DqnAgent(self.step_spec,self.action_spec,q_network=self.q_net,optimizer=self.optimizer,td_errors_loss_fn=common.element_wise_squared_loss,train_step_counter=self.train_step_counter)
        self.agent.initialize()

        self.save_name = self.dqn_conf['save_name']

    def act(self,obs):
        return self.agent.policy.action(obs)

    def compute_avg_return(self):
        """
        Function from https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial tutorial.
        """
        total_return = 0.0
        for _ in range(self.num_eval_episodes):

            time_step = self.eval_env.reset()
            episode_return = 0.0

            while not time_step.is_last():

                action_step = self.act(time_step)
                time_step = self.eval_env.step(action_step.action)
                episode_return += time_step.reward

            total_return += episode_return
        avg_return = total_return / self.num_eval_episodes
        return avg_return.numpy()[0]

    def collect_step(self, buffer):
        """
        Function from https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial tutorial.
        """
        time_step = self.train_env.current_time_step()
        action_step = self.agent.policy.action(time_step)
        next_time_step = self.train_env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        buffer.add_batch(traj)

    def collect_data(self, buffer):
        """
        Function from https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial tutorial.
        """
        for _ in range(self.initial_collect_steps):
            self.collect_step(buffer)

    def save_model(self, step):
        """
        Method for saving agent.
        """
        filepath = os.path.join(os.getcwd(), 'saved_models', self.save_name, '-', str(step))
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_net.get_weights(), f)

    def load_model(self, name, step):
        """
        Method for loading agent.
        """
        filepath = os.path.join(os.getcwd(), 'saved_models', name + '-' + str(step))
        with open(filepath, 'rb') as f:
            new_weights = pickle.load(f)
        self.q_net.set_weights(new_weights)

    def train(self, plot=True):
        """
        Adapted from https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial tutorial.
        """
        tf.compat.v1.enable_v2_behavior()
        self.train_env.reset()

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self.replay_buffer_max_length)
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.batch_size,
            num_steps=2).prefetch(3)
        iterator = iter(dataset)

        self.agent.train = common.function(self.agent.train)

        self.agent.train_step_counter.assign(0)

        avg_return = self.compute_avg_return()
        returns = [avg_return]

        for _ in range(self.num_iterations):
            # Collect a few steps using collect_policy and save to the replay buffer.
            self.collect_data(replay_buffer)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = self.agent.train(experience).loss

            step = self.agent.train_step_counter.numpy()

            if step % self.log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))

            if step % self.eval_interval == 0:
                self.save_model(step)
                avg_return = self.compute_avg_return()
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)
        
        if plot:
            iterations = range(0, self.num_iterations + 1, self.eval_interval)
            plt.plot(iterations, returns)
            plt.ylabel('Average Return')
            plt.xlabel('Iterations')
            plt.ylim(top=250)

    def demo(self, load_step = 0):
        """
        Demo trained or loaded agent.
        """
        time_step = self.eval_env.reset()
        score = 0.0
        if load_step:
            self.load_model(self.save_name,load_step)
        while not time_step.is_last():
            action_step = self.act(time_step)
            time_step = self.eval_env.step(action_step.action)
            score += time_step.reward
            self.eval_env.render()
            time.sleep(0.005)
        self.eval_py_env.close()
        print('\nThe agent scored {:.2f}\n'.format(score[0]))

def main():
    dqn = AtariDQN('net.config','dqn_preset.config')
    dqn.train()
    dqn.demo()

if __name__ == "__main__":
    main()