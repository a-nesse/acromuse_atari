import time
import json
import os
import pickle
import sys
import numpy as np

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.utils import common
from tf_agents.policies import epsilon_greedy_policy

#from preprocessing import suite_atari_mod as suite_atari
from preprocessing import suite_atari_dqn as suite_atari


class AtariDQN:
    """
    Class for training Deep-Q agent to play Atari games.
    """

    def __init__(self, net_conf_path='', dqn_conf_path=''):

        def _load_config(conf_path):
            assert os.path.exists(
                conf_path), 'The config file specified does not exist.'
            with open(conf_path, 'r') as f:
                conf = json.load(f)
            return conf

        self.net_conf = _load_config(net_conf_path)
        self.dqn_conf = _load_config(dqn_conf_path)

        self.env_name = self.dqn_conf['env_name']
        self.num_iterations = self.dqn_conf['num_iterations']
        self.collect_steps_per_iteration = self.dqn_conf['collect_steps_per_iteration']
        self.parallell_calls = self.dqn_conf['parallell_calls']
        self.batch_size = self.dqn_conf['batch_size']
        self.target_update = self.dqn_conf['target_update']
        self.learning_rate = self.dqn_conf['learning_rate']
        self.log_interval = self.dqn_conf['log_interval']
        self.n_eval_steps = self.dqn_conf['n_eval_steps']
        self.eval_interval = self.dqn_conf['eval_interval']

        self.train_py_env = suite_atari.load(
            environment_name=self.env_name, eval_env=False)
        self.eval_py_env = suite_atari.load(
            environment_name=self.env_name, eval_env=True)
        self.train_env = tf_py_environment.TFPyEnvironment(self.train_py_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(self.eval_py_env)

        self.obs_spec = self.train_env.observation_spec()
        self.action_spec = self.train_env.action_spec()
        self.step_spec = self.train_env.time_step_spec()

        self.q_net = q_network.QNetwork(
            self.obs_spec,
            self.action_spec,
            conv_layer_params=[tuple(c) for c in self.net_conf['conv_layer_params']],
            fc_layer_params=tuple(self.net_conf['fc_layer_params']),
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal'))

        self.optimizer = tf.compat.v1.train.RMSPropOptimizer(
            learning_rate=self.dqn_conf['learning_rate'],
            momentum=self.dqn_conf['momentum'],
            decay=self.dqn_conf['decay'],
            epsilon=self.dqn_conf['mom_epsilon'])

        self.train_step_counter = tf.Variable(0)

        # Replay buffer size & initial collect -3 due to stacking 4 frames
        self.replay_buffer_max_length = self.dqn_conf['replay_buffer_max_length']-3
        self.initial_collect = self.dqn_conf['initial_collect_frames']-3

        self.initial_epsilon = self.dqn_conf['initial_epsilon']
        self.final_epsilon = self.dqn_conf['final_epsilon']
        self.final_exploration = self.dqn_conf['final_exploration']

        self.agent = dqn_agent.DqnAgent(
            self.step_spec,
            self.action_spec,
            q_network=self.q_net,
            optimizer=self.optimizer,
            emit_log_probability=True,
            td_errors_loss_fn=common.element_wise_huber_loss,
            train_step_counter=self.train_step_counter,
            epsilon_greedy=1.0,
            target_update_period=self.target_update,
            gamma=self.dqn_conf['discount'])
        self.agent.initialize()

        self.save_name = self.dqn_conf['save_name']
        self.keep_n_models = self.dqn_conf['keep_n_models']
        self.log = {}

        self.elite_avg = (0, 0)  # elite model, score for average score
        self.elite_max = (0, 0)  # elite model, score for max score

        # epsilon-greedy eval policy as described by Mnih et.al (2015)
        self.eval_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
            policy=self.agent.policy,
            epsilon=self.dqn_conf['eval_epsilon'])

        # declaring
        self.replay_buffer = None
        self.replay_ckp = None
        self.driver = None


    def act(self, obs):
        '''
        Method for predicting action.
        Uses epsilon-greedy policy to avoid evaluation overfitting.
        '''
        return self.eval_policy.action(obs)

    
    def run_episode(self,steps):
        """
        Function for running an episode in the environment.
        Returns the score if the episode is finished without
        exceeding the number of evaluation steps.
        """
        episode_score = 0.0
        time_step = self.eval_env.reset()
        while not time_step.is_last():
            action_step = self.act(time_step)
            time_step = self.eval_env.step(action_step.action)
            episode_score += time_step.reward.numpy()[0]
            steps += 1
            if steps > self.n_eval_steps:
                return True, None, None
        return False, steps, episode_score


    def evaluate_agent(self):
        """
        Function for evaluating/scoring agent. 
        """
        steps = 0
        scores = []
        while True:
            done, steps, ep_score = self.run_episode(steps)
            if done:
                return np.average(scores), np.max(scores)
            scores.append(ep_score)


    def save_model(self, step):
        """
        Method for saving agent and deleting old agents.
        Saves both q network and target network.
        """
        filepath_q = os.path.join(
            os.getcwd(), 'saved_models_dqn', self.save_name + '-' + str(step) + '-eval')
        with open(filepath_q, 'wb') as f:
            pickle.dump(self.q_net.get_weights(), f)
        filepath_target = os.path.join(
            os.getcwd(), 'saved_models_dqn', self.save_name + '-' + str(step) + '-target')
        with open(filepath_target, 'wb') as f:
            pickle.dump(self.agent._target_q_network.get_weights(), f)
        # deleting old agents
        delete = step-(self.eval_interval*self.keep_n_models)
        if delete > 0 and self.elite_avg[0] != delete and self.elite_max[0] != delete:
            self.delete_model(delete)


    def load_model(self, step):
        """
        Method for loading q & target network.
        """
        filepath_q = os.path.join(
            os.getcwd(), 'saved_models_dqn', self.save_name + '-' + str(step) + '-eval')
        with open(filepath_q, 'rb') as f:
            new_weights = pickle.load(f)
        filepath_target = os.path.join(
            os.getcwd(), 'saved_models_dqn', self.save_name + '-' + str(step) + '-target')
        with open(filepath_target, 'rb') as f:
            new_target = pickle.load(f)
        frames = int(step*self.batch_size*4)
        scaled_epsilon = self.initial_epsilon - \
            (0.9*frames/self.final_exploration)
        self.agent.collect_policy._epsilon = max(
            self.final_epsilon, scaled_epsilon)
        self.q_net.set_weights(new_weights)
        self.agent._target_q_network.set_weights(new_target)


    def delete_model(self, step):
        """
        Function for deleting agent.
        """
        os.remove(os.path.join(os.getcwd(), 'saved_models_dqn',
                               self.save_name + '-' + str(step) + '-eval'))
        os.remove(os.path.join(os.getcwd(), 'saved_models_dqn',
                               self.save_name + '-' + str(step) + '-target'))


    def log_data(self, starttime, passed_time, step, loss, avg_score, max_score):
        """
        Function for logging training performance.
        """
        cur_time = time.time()
        train_time = cur_time - starttime + passed_time
        step = int(step)
        loss = float(loss)
        frames = step * self.batch_size * 4

        if step % self.eval_interval == 0:
            # if elite, replace and potentially delete old elite
            keep = step-(self.eval_interval*(self.keep_n_models-1))
            if avg_score > self.elite_avg[1] and step >= self.eval_interval:
                delete = self.elite_avg[0]
                self.elite_avg = (step, avg_score)
                # delete if not within keep interval
                if delete < keep and delete != 0 and delete != self.elite_max[0]:
                    self.delete_model(delete)
            if max_score > self.elite_max[1] and step >= self.eval_interval:
                delete = self.elite_max[0]
                self.elite_max = (step, max_score)
                # delete if not within keep interval
                if delete < keep and delete != 0 and delete != self.elite_avg[0]:
                    self.delete_model(delete)

        self.log[step] = [train_time, loss, avg_score, max_score, frames, self.elite_avg, self.elite_max]


    def write_log(self):
        """
        Function for writing log.
        """
        filepath = os.path.join(
            os.getcwd(), 'saved_models_dqn', self.save_name + 'log')
        with open(filepath, 'w') as f:
            json.dump(self.log, f)


    def load_log(self, step):
        """
        Function for loading log.
        """
        filepath = os.path.join(
            os.getcwd(), 'saved_models_dqn', self.save_name + 'log')
        with open(filepath, 'r') as f:
            log = json.load(f)
        self.log = log
        self.elite_avg = (log[str(step)][5][0], log[str(step)][5][1])
        self.elite_max = (log[str(step)][6][0], log[str(step)][6][1])


    def restart_training(self, step):
        """
        Function for restarting training from step.
        """
        self.load_model(step)
        self.load_log(step)


    def train(self, restart_step=0):
        """
        Method for running training of DQN model.
        """
        tf.compat.v1.enable_v2_behavior()
        time_step = self.train_env.reset()

        start_time = time.time()

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self.replay_buffer_max_length)

        self.replay_ckp = common.Checkpointer(
            ckpt_dir=os.path.join(
                os.getcwd(), 'saved_models_dqn', self.save_name + 'replay'),
            max_to_keep=1,
            replay_buffer=self.replay_buffer)

        # initializing dynamic step driver
        self.driver = dynamic_step_driver.DynamicStepDriver(
            self.train_env,
            self.agent.collect_policy,
            observers=[self.replay_buffer.add_batch],
            num_steps=self.collect_steps_per_iteration)

        self.driver.run = common.function(self.driver.run)

        if restart_step:
            self.restart_training(restart_step)
            self.agent.train_step_counter.assign(restart_step)
            passed_time = self.log[str(restart_step)][0]
            policy_state = self.agent.collect_policy.get_initial_state(
                self.train_env.batch_size)
        else:
            # setting epsilon to 1.0 for initial collection (random policy)
            self.agent.collect_policy._epsilon = self.initial_epsilon
            policy_state = self.agent.collect_policy.get_initial_state(
                self.train_env.batch_size)
            for _ in range(self.initial_collect):
                time_step, policy_state = self.driver.run(
                    time_step=time_step,
                    policy_state=policy_state)
            self.agent.train_step_counter.assign(0)
            passed_time = 0

        self.replay_ckp.initialize_or_restore()
        # saving initial buffer to make sure that memory is sufficient
        self.replay_ckp.save(global_step=restart_step)

        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=self.parallell_calls,
            sample_batch_size=self.batch_size,
            num_steps=2).prefetch(self.parallell_calls)
        iterator = iter(dataset)

        self.agent.train = common.function(self.agent.train)

        # eval before training
        if restart_step:
            avg_score = self.log[str(restart_step)][2]
            max_score = self.log[str(restart_step)][3]
        else:
            avg_score, max_score = self.evaluate_agent()

        exploration_finished = False

        for _ in range(self.num_iterations-restart_step):

            # performing action according to epsilon-greedy protocol & collecting data
            time_step, policy_state = self.driver.run(
                time_step=time_step,
                policy_state=policy_state)

            # sampling from data
            experience, unused_info = next(iterator)

            # training
            train_loss = self.agent.train(experience).loss
            step = self.agent.train_step_counter.numpy()

            frames = int(step*self.batch_size*4)
            # changing epsilon linearly from frames 0 to 1 mill, down to 0.1
            if frames <= self.final_exploration:
                scaled_epsilon = self.initial_epsilon - \
                    (0.9*frames/self.final_exploration)
                self.agent.collect_policy._epsilon = max(
                    self.final_epsilon, scaled_epsilon)
            elif not exploration_finished:
                self.agent.collect_policy._epsilon = self.final_epsilon
                exploration_finished = True

            if step % self.eval_interval == 0 and step != restart_step:
                self.save_model(step)
                self.replay_ckp.save(global_step=step)
                avg_score, max_score = self.evaluate_agent()
                print('step = {}: Average Score = {} Max Score = {}'.format(
                    step, avg_score, max_score))

            if step % self.log_interval == 0:
                print(time.time()-start_time)
                self.log_data(start_time, passed_time, step,
                              train_loss, avg_score, max_score)
                if step % self.eval_interval == 0:
                    self.write_log()
                print('step = {}: loss = {}'.format(step, train_loss))


def main(step):
    '''
    Creates AtariDQN object and runs training according to configs.
    '''
    net_conf = os.path.abspath(os.path.join('..', 'configs', 'net.config'))
    dqn_conf = os.path.abspath(os.path.join(
        '..', 'configs', 'dqn_preset.config'))
    dqn = AtariDQN(net_conf, dqn_conf)
    if not os.path.isdir(os.path.join(os.getcwd(), 'saved_models_dqn')):
        os.makedirs(os.path.join(os.getcwd(), 'saved_models_dqn'))
    dqn.train(step)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        main(int(args[0]))
    else:
        main(0)
