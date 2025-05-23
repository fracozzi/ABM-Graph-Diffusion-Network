from models.surrogate.nnmodel import NNModel
from models.surrogate.nnmodel_ablation import NNModel_ablation
from models.abm.predpreyfeaturizer import PredPreyFeaturizer

from config import configuration

import pickle

def main():

    # Load the configuration
    abm_model = configuration['abm_model']
    model_type = configuration['model_type']
    learning_rate = configuration['learning_rate']
    T_diffusion = configuration['T_diffusion']
    n_epochs = configuration['n_epochs']

    # Load the training data
    with open(f'../../ramifications/{abm_model}/ramification_training.pickle', 'rb') as f:
        ramification_training = pickle.load(f)

    # Load the testing data
    with open(f'../../ramifications/{abm_model}/ramification_testing.pickle', 'rb') as f:
        ramification_testing = pickle.load(f)

    # Initialize the model
    if model_type == 'surrogate':
        model = NNModel(learning_rate=learning_rate, T_diffusion=T_diffusion, n_epochs=n_epochs)
    elif model_type == 'ablation':
        model = NNModel_ablation(learning_rate=learning_rate, T_diffusion=T_diffusion, n_epochs=n_epochs)
    
    # Train the model
    model.train(ramification_training)

    # Test the model
    model.test(ramification_testing)

if __name__ == "__main__":
    main()