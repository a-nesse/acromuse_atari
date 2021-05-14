import pickle
import json
import os
import sys
import gc
import time

from atari_evolution import AtariEvolution
from dqn_implementation.atari_dqn import AtariDQN

class combine_GA_DQN:
    """
    This class runs the ACROMUSE genetic algorithm initially.
    Then switches to using DQN for training agent.
    """

    def __init__(self,n_ga_gens,evo_folder,dqn_folder,net_conf_path,dqn_conf_path,evo_conf_path):
        self.n_ga_gens = n_ga_gens
        self.final_gen = n_ga_gens-1
        self.evo_folder = evo_folder
        self.dqn_folder = dqn_folder
        self.net_conf_path = net_conf_path
        self.dqn_conf_path = dqn_conf_path
        self.evo_conf_path = evo_conf_path

        def _load_config(conf_path):
            try:
                assert os.path.exists(conf_path)
            except IOError:
                print('The config file specified does not exist.')
            with open(conf_path, 'r') as f:
                conf = json.load(f)
            return conf
        
        self.evo_conf = _load_config(evo_conf_path)
        self.dqn_conf = _load_config(dqn_conf_path)
        self.evo_save_name = self.evo_conf['save_name']
        self.dqn_save_name = self.dqn_conf['save_name']

        #generating directories
        if not os.path.isdir(dqn_folder):
            os.makedirs(dqn_folder)
        if not os.path.isdir(evo_folder):
            os.makedirs(evo_folder)

        self.init_time = 0


    def _initialize_dqn_experience(self):
        """
        Initializes the DQN experience replay buffer.
        This is saved to disk and the agent is deleted to free up memory.
        This is done so that the DQN agent can use the 'restart' procedure 
        once the GA is done.
        """
        # initialize experience replay buffer
        init_start_time = time.time() # timing the initialization
        exp_init = AtariDQN(net_conf_path=self.net_conf_path,dqn_conf_path=self.dqn_conf_path)
        exp_init.num_iterations = 0 # no training
        exp_init.train() # buffer is saved to disk
        self.init_time = init_start_time-time.time()
        # deleting object & freeing up memory
        del exp_init
        gc.collect()


    def _make_log_entry(self,gen_list):
        """
        Creates DQN log entry from ACROMUSE log entry.
        """
        time = gen_list[0] + self.init_time # add time to set up experience buffer
        elite_avg_score = gen_list[2]
        elite_max_score = gen_list[3]
        entry = [time,0,elite_avg_score,elite_max_score,0,[0,0],[0,0]]
        return entry


    def _retrieve_evo_log(self):
        """
        Retrieves ACROMUSE log
        """
        with open(os.path.join(self.evo_folder,self.evo_save_name + '-log'),'r') as f:
            evo_log = json.load(f)
        return evo_log
    

    def _write_dqn_log(self,dqn_log):
        """
        Writes initial DQN log
        """
        with open(os.path.join(self.dqn_folder,self.dqn_save_name + 'log'), 'w') as f:
            json.dump(dqn_log,f)


    def make_dqn_log(self):
        """
        Makes initial DQN log based on ACROMUSE log to facilitate DQN start.
        """
        evo_log = self._retrieve_evo_log()
        dqn_log = {}
        dqn_log[str(1)] = self._make_log_entry(evo_log[str(self.final_gen)])
        self._write_dqn_log(dqn_log)


    def _find_elite(self):
        """
        Finds elite agent in final generation.
        """
        with open(os.path.join(self.evo_folder,self.evo_save_name + '-elites'),'r') as f:
            elites = json.load(f)
        final_elite = elites[str(self.final_gen)]
        return final_elite


    def _retrieve_evo_network(self):
        """
        Retrieves network after evolution.
        """
        elite = self._find_elite()
        path = os.path.join(self.evo_folder,self.evo_save_name + '-' + str(self.final_gen) + '-' + str(elite))
        with open(path,'rb') as f:
            network = pickle.load(f)
        return network


    def place_network(self):
        """
        Writes Q-network and target network to start DQN training.
        """
        network = self._retrieve_evo_network()
        path_e = os.path.join(self.dqn_folder, self.dqn_save_name + '-' + str(1) + '-eval')
        path_t = os.path.join(self.dqn_folder, self.dqn_save_name + '-' + str(1) + '-target')
        paths = [path_e,path_t]
        for path in paths:
            with open(path,'wb') as f:
                pickle.dump(network,f)


    def run(self):
        """
        Runs training with ACROMUSE and transitions to DQN.
        """
        # initializing DQN expeirence buffer
        self._initialize_dqn_experience()

        # initial evolution
        evolver = AtariEvolution(self.net_conf_path,self.evo_conf_path)
        if not os.path.isdir(os.path.join(os.getcwd(),'saved_models_evo')):
            os.makedirs(os.path.join(os.getcwd(),'saved_models_evo'))
        evolver.n_gens = self.n_ga_gens # setting max generations
        evolver.evolve(0)

        # preparing for DQN
        self.make_dqn_log()
        self.place_network()

        # deleting evolver & freeing up memory
        del evolver
        gc.collect()

        # starting DQN training
        dqn = AtariDQN(net_conf_path=self.net_conf_path, dqn_conf_path=self.dqn_conf_path)
        dqn.train(restart_step=1)


    def restart(self, restart_step):
        """
        Restarts training at restart_step. 
        Is only capable of restarting once DQN training has started.
        """
        dqn = AtariDQN(net_conf_path=self.net_conf_path, dqn_conf_path=self.dqn_conf_path)
        dqn.train(restart_step=restart_step)


def main(restart_step):
    """
    Main function runs combined training and defines necessary parameters.
    """
    n_ga_gens = 5
    evo_folder = os.path.join(os.getcwd(),'saved_models_evo')
    dqn_folder = os.path.join(os.getcwd(),'dqn_implementation','saved_models_dqn')
    conf_path = os.path.join(os.getcwd(),'configs')
    net_conf_path = os.path.join(conf_path, 'net.config')
    dqn_conf_path = os.path.join(conf_path, 'dqn.config')
    evo_conf_path=os.path.join(conf_path,'acromuse.config')
    trainer = combine_GA_DQN(n_ga_gens,evo_folder,dqn_folder,net_conf_path,dqn_conf_path,evo_conf_path)
    if restart_step:
        trainer.restart(restart_step)
    else:
        trainer.run()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args)==1:
        main(int(args[0]))
    else:
        main(0)