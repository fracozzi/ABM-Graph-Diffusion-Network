from models.abm.abm import ABM, ABMFeaturizer

import numpy as np

import argparse
import pickle
import random
import tqdm

SCHELLING_STATE_VARIABLES = ['xcor_turtles', 'ycor_turtles', 'color_turtles']
grid_max_val = 25

### Utilities ###
is_neighbor = lambda x1, x2: (
        abs(x1 - x2) <= 1 or
        (x1 == grid_max_val and x2 == -grid_max_val) or
        (x1 == -grid_max_val and x2 == grid_max_val))

toroid = lambda x: (x if x <= grid_max_val else (x - 2 * grid_max_val)
                   ) if x >= -grid_max_val else (x + 2 * grid_max_val)
#def move_agent(i, x, y, max_dist=(grid_max_val * 2 // 5), max_trials=10):
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate a ramification data set with Schelling ABM')
    parser.add_argument('--density', type=float, default=0.75, help='density parameter (default: 0.5)')
    parser.add_argument('--threshold', type=float, default=0.875, help='threshold parameter (default: 0.8)')
    parser.add_argument('--n_timesteps', type=int, default=10, help='number of time steps (default: 10)')
    parser.add_argument('--n_ramifications', type=int, default=500, help='number of ramifications (default: 100)')
    parser.add_argument('--seed', type=int, default=42, help='seed value (default: 42)')
    parser.add_argument('--output_path', type=str, default='/data/fcozzi/abm-diffusion/Schelling/final_experiment/ramifications_xi2.pickle',
                        help='output file path (default: /data/fcozzi/abm-diffusion/Schelling/ramifications.pickle)')
    args = parser.parse_args()

    # Run SchellingABM
    
    params = {'xi1':0.625,
              'xi2':0.875,
              'xi3':0.75}
    for param in params.keys():

        #with open(f'/data/fcozzi/abm-diffusion/Schelling/final_experiment/ramifications_{param}.pickle','rb') as f:
        #    ramification = pickle.load(f)

        #schelling_abm = SchellingABM(density=args.density, threshold=args.threshold)
        schelling_abm = SchellingABM(density=0.75, threshold=params[param])
        with open(f'/data/fcozzi/abm-diffusion/Schelling/final_experiment/ramifications_{param}.pickle','rb') as f:
            output = pickle.load(f)

        #output = schelling_abm.generate_ramifications(
        #                            n_timesteps=args.n_timesteps, 
        #                            n_ramifications=args.n_ramifications, 
        #                            seed=random.randint(0,10000))

        future_output = schelling_abm.generate_ramifications_from_initial_condition(
            initial_condition=output[-1][0],
            n_timesteps=args.n_timesteps,
            n_ramifications=args.n_ramifications,
            seed=random.randint(0,10000)
        )
    # Save output to a pickle file
        #with open(f'/data/fcozzi/abm-diffusion/Schelling/final_experiment/ramifications_{param}.pickle', 'wb') as f:
        #    pickle.dump(output, f)
        
        with open(f'/data/fcozzi/abm-diffusion/Schelling/final_experiment/future_ramifications_{param}.pickle', 'wb') as f:
            pickle.dump(future_output, f)
        
        

if __name__ == "__main__":
    main()