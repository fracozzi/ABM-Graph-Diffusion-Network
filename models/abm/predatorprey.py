import sys
sys.path += ['../../']
import models.abm.pp_functions as pp
from models.abm.abm import ABM

import numpy as np
import pickle
import torch
from einops import rearrange
import random
import os


VARIABLES = ['KIND','PHASE','POSITION']
GRID_SIZE = 32

class PredatorPreyABM(ABM):

    def __init__(self,Psi,n_agents=1500,max_agent=1500,grid_fill_p=0.3,predators_p=0.35,grid_size=50):

        """ Initialize parameters. """        

        self.Psi = Psi
        self.n_agents = n_agents
        self.max_agents = max_agent
        self.grid_fill_p = grid_fill_p
        self.predators_p = predators_p
        self.grid_size = grid_size

        self.adjacency_set = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=torch.int8)
        self.adjacency_parent_set = torch.cat((self.adjacency_set, torch.tensor([[0, 0]], dtype=torch.int8)), axis=0)
        self.A_set = pp.generate_action_set()
        

    def initial_state(self) -> np.ndarray:
        # CONSTANTS
        
        # INITIALIZATION
        N = pp.number_initialized_agents(self.grid_fill_p, self.grid_size)
        N_pred = pp.number_initialized_predators(N, self.predators_p)
        K, S, X, NO, _, parental = pp.init_abm(self.Psi, self.n_agents, N, N_pred, self.grid_size, self.A_set)

        K = K.unsqueeze(-1)
        NO = NO.unsqueeze(-1)
        S = rearrange(S,'v a -> a v')

        initial_state = torch.hstack((K,S,X,NO,parental))
        
        return initial_state.numpy()
    
    def next_step(self, prev_state: np.ndarray) -> np.ndarray:

        prev_state = torch.tensor(prev_state)
        K = prev_state[:,0].squeeze()
        S = rearrange(prev_state[:,1:5], 'a v -> v a')
        X = prev_state[:,[5,6]]
        NO = prev_state[:,-3].squeeze()
        parental = prev_state[:,-2:]
        
        A = pp.evolve_A(self.Psi, K, S, NO, self.A_set)
        
        S_new, X_new, NO_new , _ = pp.evolve_one_step(K, X, A, parental, self.Psi, self.grid_size, self.A_set, self.adjacency_set, self.adjacency_parent_set)

        parental_new = pp.update_parental(parental,K,S_new)

        K = K.unsqueeze(-1)
        NO_new = NO_new.unsqueeze(-1)
        S_new = rearrange(S_new,'v a -> a v')

        next_state = torch.hstack((K,S_new,X_new,NO_new,parental_new))

        return next_state.numpy()


def main():

    # Specify ABM parameters

    # Probabilities = [TO DIE, TO MOVE, TO REPRODUCE,ALREADY PREGNANT, ALREADY DEAD, NOT BORN YET]
    
    Psi_1 = torch.tensor([[0.15, 0.45, 0.4, 0.0, 0.0, 0.0],      #pred+prey
                            [0.25, 0.55, 0.2, 0.0, 0.0, 0.0],    #pred+noprey 
                            [0.3, 0.45, 0.25, 0.0, 0.0, 0.0],    #prey+pred
                            [0.15, 0.4, 0.45, 0.0, 0.0, 0.0],    #prey+nopred
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],      #pregnant
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],      #dead
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])     #unborn
    
    Psi_2 = torch.tensor([[0.35,0.45,0.2,0.0,0.0,0.0],           #pred+prey
                            [0.25,0.6,0.15,0.0,0.0,0.0],         #pred+noprey
                            [0.45,0.5,0.05,0.0,0.0,0.0],         #prey+pred
                            [0.35,0.35,0.3,0.0,0.0,0.0],         #prey+nopred
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],      #pregnant
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],      #dead
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])     #unborn
    
    Psi_3 = torch.tensor([[0.15,0.3,0.55,0.0,0.0,0.0],           #pred+prey
                            [0.3,0.55,0.15,0.0,0.0,0.0],         #pred+noprey
                            [0.7,0.2,0.10,0.0,0.0,0.0],          #prey+pred
                            [0.1,0.4,0.5,0.0,0.0,0.0],           #prey+nopred
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],      #pregnant
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],      #dead
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])     #unborn
    
    Psi_4 = torch.tensor([[0.15,0.35,0.5,0.0,0.0,0.0],           #pred+prey
                            [0.25,0.45,0.3,0.0,0.0,0.0],         #pred+noprey
                            [0.45,0.4,0.15,0.0,0.0,0.0],         #prey+pred
                            [0.3,0.4,0.3,0.0,0.0,0.0],           #prey+nopred
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],      #pregnant
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],      #dead
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])     #unborn

    Psi_set = [Psi_1,Psi_2,Psi_3,Psi_4]
    M = 2*GRID_SIZE*GRID_SIZE
    N_MAX = GRID_SIZE*GRID_SIZE
    grid_fill_p = 0.3
    predators_p = 0.5

    # Ramification specifications

    timesteps_training = 3
    timesteps_testing = 3
    n_ramifications = 5

    names = ['psi1','psi2','psi3','psi4']
    base_dir = '../../ramifications'
    sub_dir = 'predatorprey'
    save_dir = os.path.join(base_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, Psi in enumerate(Psi_set):

        # Create ramification for training

        predatorprey_abm = PredatorPreyABM(Psi=Psi,n_agents=M,max_agent=N_MAX,
                                    grid_fill_p=grid_fill_p,predators_p=predators_p,
                                    grid_size=GRID_SIZE)

        ramification_training = predatorprey_abm.generate_ramifications(n_timesteps=timesteps_training,n_ramifications=n_ramifications,
                                                               seed=random.randint(0,10000))
        
        
        path = os.path.join(save_dir, f'ramification_training_{names[i]}.pickle')
        with open(path,'wb') as f:
            pickle.dump(ramification_training,f)

        # Create ramification for testing
        initial_condition = ramification_training[-1][0]
        ramification_testing = predatorprey_abm.generate_ramifications_from_initial_condition(
            initial_condition=initial_condition,
            n_timesteps=timesteps_testing,
            n_ramifications=n_ramifications,
            seed=random.randint(0,10000)
        )

        #Save output to a pickle file
        path = os.path.join(save_dir, f'ramification_testing_{names[i]}.pickle')
        with open(path, 'wb') as f:
            pickle.dump(ramification_testing, f)
        

if __name__ == "__main__":
    main()
