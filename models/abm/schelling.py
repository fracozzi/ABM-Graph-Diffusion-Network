import numpy as np
import pickle
import random
import os
import pyrootutils

ROOT_DIR = pyrootutils.setup_root(__file__, indicator="README.md", pythonpath=True)

from models.abm.abm import ABM

RAMIFICATIONS_DIR = os.path.join(ROOT_DIR, "ramifications")
if not os.path.exists(RAMIFICATIONS_DIR):
    os.makedirs(RAMIFICATIONS_DIR)
os.chdir(ROOT_DIR)

SCHELLING_STATE_VARIABLES = ['xcor_turtles', 'ycor_turtles', 'color_turtles']
grid_max_val = 25

### Utilities ###
is_neighbor = lambda x1, x2: (
        abs(x1 - x2) <= 1 or
        (x1 == grid_max_val and x2 == -grid_max_val) or
        (x1 == -grid_max_val and x2 == grid_max_val))

toroid = lambda x: (x if x <= grid_max_val else (x - 2 * grid_max_val)
                   ) if x >= -grid_max_val else (x + 2 * grid_max_val)
def move_agent(i, x, y, max_dist=grid_max_val, max_trials=100):
    for _ in range(max_trials):
        heading = np.random.random() * 2 * np.pi
        dist = max_dist * np.random.random()
        dx, dy = dist * np.array([np.cos(heading), np.sin(heading)])
        xn = toroid(int(np.round(x[i] + dx)))
        yn = toroid(int(np.round(y[i] + dy)))
        if not np.any((xn == x[i]) & (yn == y[i])):
            return xn, yn
    return x[i], y[i]

################

class SchellingABM(ABM):
    """ Implementation of the Schelling segregation ABM. """

    n_state_variables = len(SCHELLING_STATE_VARIABLES)

    def __init__(self, n_agents=100, density=0.5, threshold=42):
        """ Initialize parameters. """
        self.threshold = threshold
        self.density = density
        self.n_agents = int(self.density * (grid_max_val * 2 + 1) ** 2)

    def initial_state(self) -> np.ndarray:
        """ Create a matrix representing the state at t=0. """
        return np.vstack([
            np.random.randint(-grid_max_val, grid_max_val, size=self.n_agents), # X
            np.random.randint(-grid_max_val, grid_max_val, size=self.n_agents), # Y
            np.random.randint(0, 2, size=self.n_agents), # Color
        ]).T

    def next_step(self, prev_state: np.ndarray) -> np.ndarray:
        """ Given a matrix representing the state at t, 
            returns a matrix representing state at t+1.
        """
        assert prev_state.shape == (self.n_agents, self.n_state_variables)
        n_agents = len(prev_state)
        x, y, col = prev_state.T
        assert np.min(x) >= -grid_max_val and np.max(x) <= grid_max_val
        assert np.min(y) >= -grid_max_val and np.max(y) <= grid_max_val
        xn, yn = x.copy(), y.copy()
        for i in range(n_agents):
            similar2count = {True: 0, False: 0}
            for j in range(n_agents):
                if i != j and (
                    (x[i] == x[j] and is_neighbor(y[i], y[j])) or
                    (y[i] == y[j] and is_neighbor(x[i], x[j]))
                ):
                    similar2count[col[i] == col[j]] += 1
        
            if sum(similar2count.values()) > 0:
                frac_similar = similar2count[True] / (similar2count[False] + similar2count[True])
                if frac_similar < self.threshold:
                    xn[i], yn[i] = move_agent(i, x, y)
            
        next_state = np.vstack([xn, yn, col]).T
        assert next_state.shape == (self.n_agents, self.n_state_variables)
        return next_state

def main():

    # Run SchellingABM
    
    params = {'xi1':0.625,
              'xi3':0.75,
              'xi2':0.875
              }
    
    grid_density = 0.75
    timesteps_training = 10
    timesteps_testing = 25
    n_ramifications = 500

    sub_dir = 'schelling'
    save_dir = os.path.join(RAMIFICATIONS_DIR, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for param in params.keys():
    
        schelling_abm = SchellingABM(density=grid_density, threshold=params[param])

        # Generate ramification for training

        ramification_training = schelling_abm.generate_ramifications(
                                    n_timesteps=timesteps_training, 
                                    n_ramifications=n_ramifications, 
                                    seed=random.randint(0,10000))
        
        #Save output to a pickle file
        path = os.path.join(save_dir, f'ramification_training_{param}.pickle')
        with open(path, 'wb') as f:
            pickle.dump(ramification_training, f)

        initial_condition = ramification_training[-1][0]
        # Generate ramification for testing

        ramification_testing = schelling_abm.generate_ramifications_from_initial_condition(
            initial_condition=initial_condition,
            n_timesteps=timesteps_testing,
            n_ramifications=n_ramifications,
            seed=random.randint(0,10000)
        )
    #
        #Save output to a pickle file
        path = os.path.join(save_dir, f'ramification_testing_{param}.pickle')
        with open(path, 'wb') as f:
            pickle.dump(ramification_testing, f)
        

if __name__ == "__main__":
    main()