
import numpy as np
from tqdm import tqdm

from abc import ABC, abstractmethod

class ABM(ABC):
    """ This is an interface that all ABM-like classes must implement. See SchellingABM or PredatorPreyABM. """
    @abstractmethod
    def initial_state(self):
        """ Create a matrix representing the state at t=0. """
        pass

    @abstractmethod
    def next_step(self, state: np.ndarray) -> np.ndarray:
        """ Given a matrix representing the state at t, 
            returns a matrix representing state at t+1.
        """
        pass

    def generate_ramifications(self, n_timesteps: int, n_ramifications: int, seed: int):
        """ Given an ABM, returns a ramification data set. The ramification data set is built as
            a list, where the t-th element of the list represent all the states generated for
            timestep t. Each of the states is generated from the 0-th state of the previous time step.
        """
        np.random.seed(seed)
        timesteps = []
        main_next_state = self.initial_state()
        timesteps.append([main_next_state])
        for t in tqdm(range(1, n_timesteps), desc="Timesteps"):
            runs = []
            previous_state = main_next_state
            main_next_state = None
            #for run in tqdm(range(n_ramifications), desc="Ramifications"):
            for run in range(n_ramifications):
                state = self.next_step(previous_state)
                runs.append(state)
                if main_next_state is None:
                    main_next_state = state
            timesteps.append(runs)
        return timesteps
    
    def generate_ramifications_from_initial_condition(self, initial_condition, n_timesteps: int, n_ramifications: int, seed: int):
        """ Given an ABM, returns a ramification data set. The ramification data set is built as
            a list, where the t-th element of the list represent all the states generated for
            timestep t. Each of the states is generated from the 0-th state of the previous time step.
        """
        np.random.seed(seed)
        timesteps = []
        main_next_state = initial_condition
        timesteps.append([main_next_state])
        for t in tqdm(range(1, n_timesteps), desc="Timesteps"):
            runs = []
            previous_state = main_next_state
            main_next_state = None
            for run in range(n_ramifications):
                state = self.next_step(previous_state)
                runs.append(state)
                if main_next_state is None:
                    main_next_state = state
            timesteps.append(runs)
        return timesteps
    
class ABMFeaturizer(ABC):
    @abstractmethod
    def get_shape_state_features(self,state: np.array) -> tuple:
        """ Given the whole state give shape of state variables
            and features. Features are assumed to be dicrete values
        
        """

    @abstractmethod
    def scale_abm_state(self, state: np.ndarray) -> np.ndarray:
        """ Translates a state of the ABM into an input suitable for a NN.
            Continuous variables are scaled between [0, 1].
            Discrete variables are represented through one hot encoding.
        """
        pass

    @abstractmethod
    def unscale_abm_state(self, state: np.ndarray) -> np.ndarray:
        """ Translates back the output of `scale_abm_state` into a state of the ABM.
        """
        pass

    @abstractmethod
    def get_interaction_graph(self, state: np.ndarray) -> np.ndarray:
        """ Given the current state of the ABM, extracts the
            graph of who might interact with whom.
            Such a graph is returned as an edge_index matrix, i.e.
            an 2x|E| list of edges, where column (i, j) means that
            in this state agent i could affect agent j.
        """
        pass

