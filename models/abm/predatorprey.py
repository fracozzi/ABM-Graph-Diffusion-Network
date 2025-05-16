import models.abm.pp_functions as pp
from models.abm.abm import ABM, ABMFeaturizer

import numpy as np
import pandas
import pickle
import torch
from einops import rearrange
import argparse
import random

VARIABLES = ['KIND','PHASE','POSITION']
GRID_SIZE = 32

class PredatorPreyABM(ABM):

    def __init__(self,Z,n_agents=1500,max_agent=1500,grid_fill_p=0.3,predators_p=0.35,grid_size=50):
        
        #if grid_size*grid_size < n_agents:
        #    raise "Number agents exceeds grid size available"
        
        self.Z = Z
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
        K, S, X, NO, _, parental = pp.init_abm(self.Z, self.n_agents, N, N_pred, self.grid_size, self.A_set)

        K = K.unsqueeze(-1)
        NO = NO.unsqueeze(-1)
        #X = X.numpy()
        S = rearrange(S,'v a -> a v')

        initial_state = torch.hstack((K,S,X,NO,parental))
        
        return initial_state.numpy()
    
    def next_step(self, prev_state: np.ndarray) -> np.ndarray:

        #K = torch.tensor(prev_state[:,0],dtype=torch.int8).squeeze()
        prev_state = torch.tensor(prev_state)
        K = prev_state[:,0].squeeze()
        #S = rearrange(torch.tensor(prev_state[:,1:4], dtype = torch.int8), 'a v -> v a')
        S = rearrange(prev_state[:,1:5], 'a v -> v a')
        #X = torch.tensor(prev_state[:,[4,5]])
        X = prev_state[:,[5,6]]
        #NO = torch.tensor(prev_state[:,-1]).squeeze()
        NO = prev_state[:,-3].squeeze()
        parental = prev_state[:,-2:]
        
        
        #A = rearrange(torch.tensor(prev_state[:,7:13], dtype=torch.int8), 'a v -> v a')
        A = pp.evolve_A(self.Z, K, S, NO, self.A_set)
        
        S_new, X_new, NO_new , _ = pp.evolve_one_step(K, X, A, parental, self.Z, self.grid_size, self.A_set, self.adjacency_set, self.adjacency_parent_set)

        parental_new = pp.update_parental(parental,K,S_new)

        K = K.unsqueeze(-1)
        NO_new = NO_new.unsqueeze(-1)
        #X_new = X_new.numpy()
        S_new = rearrange(S_new,'v a -> a v')
        #A_new = rearrange(A,'v a -> a v').numpy()

        next_state = torch.hstack((K,S_new,X_new,NO_new,parental_new))

        return next_state.numpy()


def main():

    # Specify ABM parameters

    # Probabilities = [TO DIE, TO MOVE, TO REPRODUCE,ALREADY PREGNANT, ALREADY DEAD, NOT BORN YET]
    
    Psi_1 = torch.tensor([[0.15, 0.45, 0.4, 0.0, 0.0, 0.0],   #pred+prey
                            [0.25, 0.55, 0.2, 0.0, 0.0, 0.0],    #pred+noprey 
                            [0.3, 0.45, 0.25, 0.0, 0.0, 0.0],   #prey+pred
                            [0.15, 0.4, 0.45, 0.0, 0.0, 0.0],   #prey+nopred
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],     
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]) 
    Psi_2 = torch.tensor([[0.35,0.45,0.2,0.0,0.0,0.0],       #pred+prey
                            [0.25,0.6,0.15,0.0,0.0,0.0],       #pred+noprey
                            [0.45,0.5,0.05,0.0,0.0,0.0],       #prey+pred
                            [0.35,0.35,0.3,0.0,0.0,0.0],       #prey+nopred
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    Psi_3 = torch.tensor([[0.15,0.3,0.55,0.0,0.0,0.0],       #pred+prey
                            [0.3,0.55,0.15,0.0,0.0,0.0],       #pred+noprey
                            [0.7,0.2,0.10,0.0,0.0,0.0],       #prey+pred
                            [0.1,0.4,0.5,0.0,0.0,0.0],       #prey+nopred
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    Psi_4 = torch.tensor([[0.15,0.35,0.5,0.0,0.0,0.0],       #pred+prey
                            [0.25,0.45,0.3,0.0,0.0,0.0],       #pred+noprey
                            [0.45,0.4,0.15,0.0,0.0,0.0],       #prey+pred
                            [0.3,0.4,0.3,0.0,0.0,0.0],       #prey+nopred
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    Psi_set = [Z_1,Z_2,Z_3,Z_4]
    M = 2*GRID_SIZE*GRID_SIZE
    N_MAX = GRID_SIZE*GRID_SIZE
    grid_fill_p = 0.3
    predators_p = 0.5

    # Ramification specifications

    n_timesteps = 10 # Number of timestep main branch of ramification
    n_ramifications = 500 # Number of ramifications 

    save_dir = '/dir/to/save/'

    names = ['Z1','Z2','Z3','Z4']

    for i, Z in enumerate(Z_set):

        predatorprey_abm = PredatorPreyABM(Z=Z,n_agents=M,max_agent=N_MAX,
                                    grid_fill_p=grid_fill_p,predators_p=predators_p,
                                    grid_size=GRID_SIZE)

        ramification = predatorprey_abm.generate_ramifications(n_timesteps=n_timesteps,n_ramifications=n_ramifications,
                                                               seed=random.randint(0,10000))
        
        path = os.path.join(save_dir, f'ramification_{names[i]}.pickle')
        with open(path,'wb') as f:
            pickle.dump(ramification,f)
            

if __name__ == "__main__":
    main()
