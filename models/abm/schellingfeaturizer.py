from models.abm.abm import ABMFeaturizer
from models.abm.schelling import SCHELLING_STATE_VARIABLES, grid_max_val

import numpy as np
import torch

class SchellingFeaturizer(ABMFeaturizer):
    """ Class that can transform a state as produced by SchellingABM into the input of our general model.
        This class encapsulates all the assumptions made on the model by the general-purpose NN.
    """
    n_state_variables = len(SCHELLING_STATE_VARIABLES)

    def __init__(self):
        pass

    def get_shape_state_features(self) -> tuple:
        # (state dimension, number of classes for discrete feature)
        return (2,2)
    
    def scale_abm_state(self, state: np.ndarray) -> np.ndarray:
        """ Translates a state of the ABM into an input suitable for a NN.
            Continuous variables are scaled between [-1, 1].
            Discrete variables are represented through one hot encoding.
        """
        continous_variables = state[:,:-1]
        discrete_variables = state[:,-1]
        
        continous_variables = continous_variables/grid_max_val
        
        one_hot = np.empty((np.shape(discrete_variables)[0],2))
        one_hot[np.where(discrete_variables==0)] = np.array([0,1])
        one_hot[np.where(discrete_variables==1)] = np.array([1,0])
                
        return np.hstack((one_hot, continous_variables))

    def unscale_abm_state(self, state: np.ndarray) -> np.ndarray:
        """ Translates back the output of `scale_abm_state` into a state of the ABM.
        """
        domain = (-1,1)
        bin_width = 1./grid_max_val

        half_bin_width = bin_width/2
        bins = np.arange(domain[0]-half_bin_width,domain[1]+bin_width,bin_width)
        intervals = np.vstack((bins[:-1],bins[1:])).T
        mid_values = np.round((intervals.T[0]+intervals.T[1])/2,2)
        
        state_copy = state.copy()
        condition = (state_copy<=intervals[0][0])
        state_copy[condition] = mid_values[0]
        condition = (state_copy>=intervals[-1][-1])
        state_copy[condition] = mid_values[-1]
        i = 0
        for b_int in intervals:
            condition = (state>=b_int[0])&(state<b_int[1])
            state_copy[condition] = mid_values[i]
            i = i + 1    

        state_copy = state_copy*grid_max_val
        state_copy = np.round(state_copy,2)    
        return state_copy
        
    
    def get_interaction_graph(self, state: np.ndarray) -> np.ndarray:
        """ Given the current state of the ABM, extracts the
            graph of who might interact with whom.
            Such a graph is returned as an edge_index matrix, i.e.
            an 2x|E| list of edges, where column (i, j) means that
            in this state agent i could affect agent j.
        """

        assert SCHELLING_STATE_VARIABLES[0] == 'xcor_turtles'
        assert SCHELLING_STATE_VARIABLES[1] == 'ycor_turtles'
        xt, yt = state[:, [0, 1]].T
        self.n_agents = state.shape[0]
        assert np.min(xt) >= -grid_max_val and np.max(xt) <= grid_max_val
        assert np.min(yt) >= -grid_max_val and np.max(yt) <= grid_max_val
        xt = xt + grid_max_val
        yt = yt + grid_max_val
        size = grid_max_val * 2 + 1
        
        dx = np.abs(np.expand_dims(xt, -1) - np.expand_dims(xt, 0))
        dy = np.abs(np.expand_dims(yt, -1) - np.expand_dims(yt, 0))
        mask = dx > (size / 2)
        dx[mask] = size - dx[mask]
        mask = dy > (size / 2)
        dy[mask] = size - dy[mask]
        A = ((dx <= 1) & (dy <= 1)) - np.eye(self.n_agents)
        edge_index = np.vstack(np.where(A))
        return edge_index
