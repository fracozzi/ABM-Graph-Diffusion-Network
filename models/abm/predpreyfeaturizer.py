from models.abm.abm import ABMFeaturizer
from models.abm.predatorprey import VARIABLES, GRID_SIZE

import numpy as np
import torch

class PredPreyFeaturizer(ABMFeaturizer):
    """ Class that can transform a state as produced by SchellingABM into the input of our general model.
        This class encapsulates all the assumptions made on the model by the general-purpose NN.
    """
    n_state_variables = len(VARIABLES)

    def __init__(self):
        pass

    def get_shape_state_features(self) -> tuple:
        # (state dimension, number of classes for discrete feature)
        return (2,2)
    
    def ramification_scaler(self, ramification, scaler):
        dataset = np.vstack((ramification[0][0],np.vstack(ramification[1:])))
        positions = dataset[:,:,5:7]
        scaler.fit(positions)
        return scaler
    
    def scale_abm_state(self, state: np.ndarray) -> np.ndarray:
        """ Translates a state of the ABM into an input suitable for a NN.
            Continuous variables are scaled between [-1, 1].
            Discrete variables are represented through one hot encoding.
        """

        kind = state[:,0]
        phase = state[:,1:5]
        position = state[:,5:7]
        
        kind_onehot = np.empty((np.shape(kind)[0],2))
        kind_onehot[kind==0] = np.array([0,1])
        kind_onehot[kind==1] = np.array([1,0])

        
        phase = phase.astype('long')

        position[np.isnan(position)] = -1               # Assigning -1 to NaN agents (dead or unborn)
        position = position + 1
        position = ((position/(GRID_SIZE)) - 0.5) * 2

        state = np.hstack((kind_onehot,phase,position))

        return state
    
    def unscale_abm_state(self, state: np.ndarray) -> np.ndarray:
        """ Translates back the output of `scale_abm_state` into a state of the ABM.
        """

        # Binning the spatial coordinates (x,y)
        
        domain = (-1,1)
        bin_width = 1./ (GRID_SIZE)

        half_bin_width = bin_width/2
        bins = np.arange(domain[0]-half_bin_width,domain[1]+bin_width,bin_width)
        intervals = np.vstack((bins[:-1],bins[1:])).T
        mid_values = np.round((intervals.T[0]+intervals.T[1])/2,2)
        
        state_copy = state.copy()        
        condition = (state_copy[:,:,-2:]<intervals[0][0])
        state_copy[:,:,-2:][condition] = mid_values[0] # Here we deal with the NaN with value mid_values[0] = -1
        condition = (state_copy[:,:,-2:]>=intervals[-1][-1])
        state_copy[:,:,-2:][condition] = mid_values[-1]
        i = 0
        for b_int in intervals:
            condition = (state[:,:,-2:]>=b_int[0])&(state[:,:,-2:]<b_int[1])
            state_copy[:,:,-2:][condition] = mid_values[i]
            i = i + 1    

        state_copy[:,:,-2:] = (((state_copy[:,:,-2:] * 0.5) + 0.5) * ( GRID_SIZE)) - 1
        
        # Rescaling phases (ALIVE, DEAD, PREGNANT, NOT BORN)
        max_index = np.argmax(state_copy[:,:,:4],axis=-1)
        dim0_indices = np.arange(state_copy.shape[0])[:, None]  
        dim1_indices = np.arange(state_copy.shape[1])[None, :] 
        categorical = np.zeros_like(state_copy[:,:,:4])
        categorical[dim0_indices,dim1_indices,max_index] = 1
        state_copy[:,:,:4] = categorical

        state_copy = np.round(state_copy,0)  
    
        return state_copy
    
    def get_interaction_graph(self, state: np.ndarray) -> np.ndarray:
        """ Given the current state of the ABM, extracts the
            graph of who might interact with whom.
            Such a graph is returned as an edge_index matrix, i.e.
            an 2x|E| list of edges, where column (i, j) means that
            in this state agent i could affect agent j.
        """

        parental = state[:,-2:]
        graph = parental[np.logical_not(np.isnan(parental))].reshape(-1,2)
        graph = graph[np.logical_not(np.logical_or(graph[:,1] == -1,graph[:,0] == -1))]
        graph = graph.astype('int64').T

        position = state[:,[5,6]]
        position[position == -1] = np.nan               # Rescale to NaN to discard dead agents from the spatial graph
        self.n_agents = state.shape[0]
        
        delta_matrix = np.abs(position[:,None] - position)
        effective_matrix = np.minimum(delta_matrix, GRID_SIZE - delta_matrix)
        distance = np.sum(effective_matrix,axis=2)
        A = (distance <= 1)
        np.fill_diagonal(A,0)
        neighbors = np.vstack(np.where(A))
        
        edge_index = np.hstack((neighbors,graph))

        return edge_index
