"""
Here set the configuration to train the model:

abm_model = 'schelling' or 'predator_prey'
model_type = 'surrogate' or 'ablation'

configuration = {'abm_model' : 'schelling',
                 'model_type':'surrogate',
                 'learning_rate':1e-5,
                 'T_diffusion':100,
                 'n_epochs':100
                }

OR 

configuration = {'abm_model' : 'predator_prey',
                 'model_type':'ablation',
                 'learning_rate':1e-5,
                 'T_diffusion':100,
                 'n_epochs':100
                }

"""

configuration = {'abm_model' : 'predator_prey',
                 'model_type':'surrogate',
                 'learning_rate':1e-5,
                 'T_diffusion':100,
                 'n_epochs':100
                }