from models.surrogate.nnmodel import NNModel
from models.surrogate.nnmodel_ablation import NNModel_ablation
from models.abm.predpreyfeaturizer import PredPreyFeaturizer
from models.abm.schellingfeaturizer import SchellingFeaturizer

import torch
import argparse
import pickle
import os

def main():

    parser = argparse.ArgumentParser(description='Train surrogate model on an ABM')
    parser.add_argument('--abm_model', type=str, default='predatorprey', help='choose between predatorprey and schelling (default: predatorprey)')
    parser.add_argument('--model_type', type=str, default='surrogate', help='choose between surrogate and ablation (default: surrogate)')
    parser.add_argument('--parameter', type=str, default='psi1', help='choose parameter between: xi1, xi2, xi3 for schelling; and psi1, psi2, psi3, psi4 for predator-prey (default: psi1)')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate for the model (default: 1e-5)')
    parser.add_argument('--T_diffusion', type=int, default=100, help='number of diffusion steps (default: 100)')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs for training (default: 100)')  
    args = parser.parse_args()

    # Load the configuration
    abm_model = args.abm_model
    model_type = args.model_type
    parameter = args.parameter
    learning_rate = args.learning_rate
    T_diffusion = args.T_diffusion
    n_epochs = args.n_epochs


    # Load the training data
    with open(f'./ramifications/{abm_model}/ramification_training_{parameter}.pickle', 'rb') as f:
        ramification_training = pickle.load(f)


    # Initialize the model
    if abm_model == 'predatorprey':
        featurizer = PredPreyFeaturizer()
        if model_type == 'surrogate':
            model = NNModel(n_features = 6, learning_rate=learning_rate, abm_featurizer = featurizer,diffusion_timesteps=T_diffusion,aggregation='add')
        elif model_type == 'ablation':
            model = NNModel_ablation(n_features = 6, learning_rate=learning_rate, abm_featurizer = featurizer, diffusion_timesteps=T_diffusion,
                                        domain_dim=featurizer.scale_abm_state(ramification_training[0][0]).flatten().shape[0])
    elif abm_model == 'schelling':
        featurizer = SchellingFeaturizer()
        if model_type == 'surrogate':
            model = NNModel(n_features = 2, learning_rate=learning_rate, abm_featurizer = featurizer, diffusion_timesteps=T_diffusion, aggregation='mean')
        elif model_type == 'ablation':
            model = NNModel_ablation(n_features = 2, learning_rate=learning_rate, abm_featurizer = featurizer, diffusion_timesteps=T_diffusion,
                                        domain_dim=featurizer.scale_abm_state(ramification_training[0][0]).flatten().shape[0])

    model.train(ramification_training, n_epochs=n_epochs)

    # Save the model
    save_dir = f'./trained_models/{abm_model}/{model_type}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)   

    if model_type == 'surrogate':
        save_path = os.path.join(save_dir, f"model_surrogate_{parameter}.pth")
        torch.save({'ld_model_state_dict': model.ld_model.state_dict(),
                    'graph_model_state_dict': model.graph_model.state_dict(),
                    'losses': model.losses
                    }, save_path)
    
    if model_type == 'ablation':
        save_path = os.path.join(save_dir, f"model_ablation_{parameter}.pth")
        torch.save({'ld_model_state_dict': model.ld_model.state_dict(),
                    'losses': model.losses
                    }, save_path)

    print(f'Model saved to {save_path}')

if __name__ == "__main__":
    main()