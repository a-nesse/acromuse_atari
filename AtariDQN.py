import numpy as np
import gym
import time
import json
import os
#import base64
import pickle
import time
import sys
import shutil

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
#from tf_agents.environments import suite_atari
from tf_agents.drivers import dynamic_step_driver
import suite_atari_mod as suite_atari 
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
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
        self.collect_steps_per_iteration = self.dqn_conf['collect_steps_per_iteration']
        self.parallell_calls = self.dqn_conf['parallell_calls'] 
        self.batch_size = self.dqn_conf['batch_size']
        self.target_update = self.dqn_conf['target_update']
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

        self.q_net = q_network.QNetwork(self.obs_spec,self.action_spec,conv_layer_params=[tuple(c) for c in self.net_conf['conv_layer_params']],fc_layer_params=tuple(self.net_conf['fc_layer_params']),kernel_initializer=eval(self.net_conf['initializer']))
        self.optimizer = eval(self.dqn_conf['optimizer'])(learning_rate=self.dqn_conf['learning_rate'], momentum=self.dqn_conf['momentum'], decay=self.dqn_conf['discount'])
        self.train_step_counter = tf.Variable(0)

        self.replay_buffer_max_length = int(self.dqn_conf['replay_buffer_max_length']/(self.batch_size))
        self.initial_collect = self.replay_buffer_max_length

        self.agent = dqn_agent.DqnAgent(self.step_spec,self.action_spec,q_network=self.q_net,optimizer=self.optimizer,td_errors_loss_fn=common.element_wise_squared_loss,train_step_counter=self.train_step_counter,epsilon_greedy=1.0,target_update_period=self.target_update)
        self.agent.initialize()

        self.save_name = self.dqn_conf['save_name']
        self.keep_n_models = self.dqn_conf['keep_n_models']
        self.log = {}

        self.elite_avg = (0,0) #elite model, score for average score
        self.elite_max = (0,0) #elite model, score for max score


    def act(self,obs):
        return self.agent.policy.action(obs)

    def compute_avg_score(self):
        """
        Function from https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial tutorial.
        """
        total_score = 0.0
        max_score = 0.0
        for _ in range(self.num_eval_episodes):
            
            time_step = self.eval_env.reset()
            episode_score = 0.0

            while not time_step.is_last():
                action_step = self.act(time_step)
                time_step = self.eval_env.step(action_step.action)
                episode_score += time_step.reward.numpy()[0]

            total_score += episode_score
            if episode_score > max_score:
                max_score = episode_score
        avg_score = total_score / self.num_eval_episodes
        return avg_score, max_score

    def collect_step(self, buffer, policy=None):
        """
        Modified function from https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial tutorial.
        Dynamic driver?
        """
        if policy == None:
            policy = self.agent.collect_policy
        time_step = self.train_env.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = self.train_env.step(action_step)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        buffer.add_batch(traj)

    def collect_data(self, buffer, steps):
        """
        Function from https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial tutorial.
        """
        for _ in range(steps):
            self.collect_step(buffer)

    def save_model(self, step):
        """
        Method for saving agent and deleting old agents.
        Saves both q network and target network.
        """
        filepath_q = os.path.join(os.getcwd(), 'saved_models', self.save_name + '-' + str(step) + '-eval')
        with open(filepath_q, 'wb') as f:
            pickle.dump(self.q_net.get_weights(), f)
        filepath_target = os.path.join(os.getcwd(), 'saved_models', self.save_name + '-' + str(step) + '-target')
        with open(filepath_target, 'wb') as f:
            pickle.dump(self.agent._target_q_network.get_weights(), f)
        #deleting old agents
        delete = step-(self.eval_interval*self.keep_n_models)
        if delete > 0 and self.elite_avg[0]!=delete and self.elite_max[0]!=delete:
            self.delete_model(delete)

    def load_model(self, step):
        """
        Method for loading q & target network.
        """
        filepath_q = os.path.join(os.getcwd(), 'saved_models', self.save_name + '-' + str(step) + '-eval')
        with open(filepath_q, 'rb') as f:
            new_weights = pickle.load(f)
        filepath_target = os.path.join(os.getcwd(), 'saved_models', self.save_name + '-' + str(step) + '-target')
        with open(filepath_target, 'rb') as f:
            new_target = pickle.load(f)
        self.q_net.set_weights(new_weights)
        self.agent._target_q_network.set_weights(new_target)

    def delete_model(self,step):
        """
        Function for deleting magent.
        """
        os.remove(os.path.join(os.getcwd(), 'saved_models', self.save_name + '-' + str(step) + '-eval'))
        os.remove(os.path.join(os.getcwd(), 'saved_models', self.save_name + '-' + str(step) + '-target'))

    def log_data(self, starttime, passed_time, step, loss, avg_score, max_score):
        """
        Function for logging training performance.
        """
        cur_time = time.time()
        train_time = starttime - cur_time + passed_time
        frames = step * self.batch_size * 4

        #if elite, replace and potentially delete old elite
        keep = step-(self.eval_interval*self.keep_n_models)
        if avg_score > self.elite_avg[1]:
            delete = self.elite_avg[0]
            self.elite_avg = (step,avg_score)
            if keep < step and delete!=0: #delete if not within keep interval
                self.delete_model(delete)
        if max_score > self.elite_max[1]:
            delete = self.elite_max[0]
            self.elite_max = (step,max_score)
            if keep < step and delete!=0: #delete if not within keep interval
                self.delete_model(delete)

        self.log[step] = [train_time,loss,avg_score,max_score,frames,self.elite_avg,self.elite_max]

    def write_log(self,step):
        """
        Function for writing log.
        """
        filepath = os.path.join(os.getcwd(), 'logs', self.save_name + '-' + str(step))
        with open(filepath, 'wb') as f:
            pickle.dump(self.log, f)
    
    def load_log(self,step):
        """
        Function for loading log.
        """
        filepath = os.path.join(os.getcwd(), 'logs', self.save_name + '-' + str(step))
        with open(filepath, 'rb') as f:
            log = pickle.load(f)
        self.log = log

    def save_replay(self):
        """
        Function for saving replay buffer.
        """
        """
        filepath = os.path.join(os.getcwd(), 'saved_models', self.save_name + '-experience')
        with open(filepath, 'wb') as f:
            pickle.dump(self.replay_buffer, f)
        """
        pass


    def load_replay(self):
        """
        Function for loading replay.
        """
        """
        filepath = os.path.join(os.getcwd(), 'saved_models', self.save_name + '-experience')
        with open(filepath, 'rb') as f:
            replay = pickle.load(f)
        self.replay_buffer = replay
        """
        pass

    def restart_training(self, step):
        """
        Function for restarting training from step.
        """
        self.load_model(step)
        self.load_log(step)

    def train(self, restart_step=0):
        """
        Adapted & modified from https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial tutorial.
        """
        tf.compat.v1.enable_v2_behavior()
        time_step = self.train_env.reset()

        start_time = time.time()
        
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self.replay_buffer_max_length)

        self.replay_ckp = common.Checkpointer(
            ckpt_dir=os.path.join(os.getcwd(), 'saved_models', self.save_name + 'replay'),
            max_to_keep=1,
            replay_buffer = self.replay_buffer
        )

        #initializing dynamic step driver
        self.driver = dynamic_step_driver.DynamicStepDriver(
            self.train_env,
            self.agent.collect_policy,
            observers=[self.replay_buffer.add_batch],
            num_steps=self.collect_steps_per_iteration
        )
        self.driver.run = common.function(self.driver.run)

        if restart_step:
            self.restart_training(restart_step)
            self.agent.train_step_counter.assign(restart_step)
            passed_time = self.log[restart_step][0]
        else:
            policy_state = self.agent.collect_policy.get_initial_state(self.train_env.batch_size)
            for _ in range(self.initial_collect):
                time_step, policy_state = self.driver.run(
                    time_step=time_step,
                    policy_state=policy_state
                )
            self.agent.train_step_counter.assign(0)
            passed_time = 0
        
        self.replay_ckp.initialize_or_restore()

        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=self.parallell_calls,
            sample_batch_size=self.batch_size,
            num_steps=2).prefetch(self.parallell_calls)
        iterator = iter(dataset)
        
        self.agent.train = common.function(self.agent.train)

        #eval before training
        avg_score, max_score = self.compute_avg_score()

        for _ in range(self.num_iterations):

            #setting epsilon to 1.0 for initial collection (random policy)
            self.agent.collect_policy._epsilon = 1.0

            # performing action according to epsilon-greedy protocol & collecting data
            time_step, policy_state = self.driver.run(
                time_step=time_step,
                policy_state=policy_state
            )

            # sampling from data
            experience, unused_info = next(iterator)

            #training
            train_loss = self.agent.train(experience).loss
            step = self.agent.train_step_counter.numpy()

            frames = int(step*self.batch_size*4)
            #changing epsilon linearly from frames 0 to 1 mill, down to 0.1
            if frames < 1000000:
                self.agent.collect_policy._epsilon = 1.0-(0.9*frames/1000000)
                policy_state = self.agent.collect_policy.get_initial_state(self.train_env.batch_size)

            if step % self.eval_interval == 0 and step != restart_step:
                self.save_model(step)
                self.replay_ckp.save(global_step=step)
                avg_score, max_score = self.compute_avg_score()
                self.write_log(step)
                print('step = {}: Average Score = {} Max Score = {}'.format(step, avg_score, max_score))

            if step % self.log_interval == 0:
                print(time.time()-start_time)
                self.log_data(start_time,passed_time,step,train_loss,avg_score,max_score)
                print('step = {}: loss = {}'.format(step, train_loss))

def delete_archive():
    shutil.rmtree(os.path.join(os.getcwd(), 'saved_models'))
    shutil.rmtree(os.path.join(os.getcwd(), 'logs'))
    os.makedirs(os.path.join(os.getcwd(), 'saved_models'))
    os.makedirs(os.path.join(os.getcwd(), 'logs'))

def main(step, net_conf='net.config', dqn_conf='dqn_preset.config'):
    dqn = AtariDQN(net_conf,dqn_conf)
    if step == 0:
        delete_archive()
    dqn.train(step)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args)==3:
        main(args[0],args[1],args[2])
    elif len(args)==1:
        main(args[0])
    else:
        main(0)