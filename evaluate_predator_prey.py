import numpy as np
import pickle
from tqdm import tqdm
import pyemd
import torch
import random
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import gzip
import pyrootutils

ROOT_DIR = pyrootutils.setup_root(__file__, indicator="README.md", pythonpath=True)

from models.abm.abm import ABM
from models.surrogate.nnmodel import NNModel, SimulatedABM
from models.surrogate.nnmodel_ablation import NNModel_ablation, SimulatedABM_ablation1
from models.abm.predatorprey import PredatorPreyABM, GRID_SIZE
from models.abm.predpreyfeaturizer import PredPreyFeaturizer


def generate_distributions(ramifications, model):
    
    """
    Generate distributions from the surrogate model.
    """    

    surrogate_distributions = []
    for t, true_states in enumerate(ramifications[1:]):
        seed = random.randint(0,10000)
        prev_state = ramifications[t][0]
        simulated_states = model.next_step_samples(prev_state,seed = seed,n_samples=len(true_states))
        surrogate_distributions.append(simulated_states)

    surrogate_distributions = np.array(surrogate_distributions)

    return surrogate_distributions

def aggregate(forecasting_GT,forecasting_surrogate):
    tot_agents = forecasting_GT.shape[2]
    assert tot_agents == forecasting_surrogate.shape[2]
    original_prey = forecasting_GT[:,:,forecasting_GT[0][0][:,0]==0,:][:,:,:,[1,3]].sum(axis=(2,3))
    original_pred = forecasting_GT[:,:,forecasting_GT[0][0][:,0]==1,:][:,:,:,[1,3]].sum(axis=(2,3))
    diffusion_prey = forecasting_surrogate[:,:,forecasting_surrogate[0][0][:,0]==0,:][:,:,:,[1,3]].sum(axis=(2,3))
    diffusion_pred = forecasting_surrogate[:,:,forecasting_surrogate[0][0][:,0]==1,:][:,:,:,[1,3]].sum(axis=(2,3))
    delta_preys = abs(original_prey.mean(axis=0) - diffusion_prey.mean(axis=0))
    delta_preds = abs(original_pred.mean(axis=0) - diffusion_pred.mean(axis=0))
    aggregate_preys = 2 * delta_preys/ (original_prey.mean(axis=0) + diffusion_prey.mean(axis=0))
    aggregate_preds = 2 * delta_preds/ (original_pred.mean(axis=0) + diffusion_pred.mean(axis=0))
    aggregate_surrogate = 0.5 * (aggregate_preys + aggregate_preds)
    return np.array(aggregate_surrogate)

    
def micro_level_metric(ramification,distributions_surrogate):
    distributions_GT = np.array(ramification[1:])
    P = (distributions_GT[:,:,:,1:5].sum(axis=1)/distributions_GT.shape[1]).astype('float64')
    Q = (distributions_surrogate[:,:,:,:4].sum(axis=1)/distributions_surrogate.shape[1]).astype('float64')
    D = np.eye(P.shape[-1])
    D = abs(D - 1)
    emds_surrogate = [pyemd.emd(P[time,agent,:],Q[time,agent,:],D) for time in range(P.shape[0]) for agent in range(P.shape[1])]
    return np.array(emds_surrogate)



def plot_forecast(ramification,forecasting_GT,forecasting_surrogate,image_path):
    sns.set_theme(rc = {'figure.figsize' : (10,5)})
    run0 = np.array([ramification[i][0] for i in range(len(ramification))])
    starting_point_prey = run0[:,run0[0,:,0] == 0,:][:,:,[1,3]].sum(axis=(1,2))
    starting_point_pred = run0[:,run0[0,:,0] == 1,:][:,:,[1,3]].sum(axis=(1,2))
    original_prey = forecasting_GT[:,:,forecasting_GT[0][0][:,0]==0,:][:,:,:,[1,3]].sum(axis=(2,3))
    original_pred = forecasting_GT[:,:,forecasting_GT[0][0][:,0]==1,:][:,:,:,[1,3]].sum(axis=(2,3))
    diffusion_prey = forecasting_surrogate[:,:,forecasting_surrogate[0][0][:,0]==0,:][:,:,:,[1,3]].sum(axis=(2,3))
    diffusion_pred = forecasting_surrogate[:,:,forecasting_surrogate[0][0][:,0]==1,:][:,:,:,[1,3]].sum(axis=(2,3))
    colors = ['b','r']
    fig = plt.figure(figsize=(20,5))
    sns.set_style("whitegrid")
    fig.add_subplot(1,2,1)
    fig.suptitle("Forecasting")
    plt.axvline(9,ls = '--',color='g',lw=2)
    for t in original_prey:
        plt.title("Ground truth")
        plt.ylabel("Number of agents")
        plt.xlabel("Time")
        plt.plot(np.arange(0,len(ramification)),starting_point_prey,'-',color = colors[0])
        plt.plot(np.arange(len(ramification)-1,len(ramification)+forecasting_GT.shape[1]-1),np.vstack(t).T.squeeze(),"-",color=colors[0],alpha=0.5)
        plt.ylim(0,500)
    for t in original_pred:
        plt.plot(np.arange(0,len(ramification)),starting_point_pred,'-',color = colors[1])
        plt.plot(np.arange(len(ramification)-1,len(ramification)+forecasting_GT.shape[1]-1),np.vstack(t).T.squeeze(),"-",color=colors[1],alpha=0.5)
        plt.ylim(0,500)
    fig.add_subplot(1,2,2)
    plt.axvline(9,ls = '--',color='g',lw=2)
    for tt in diffusion_prey:
        plt.title("Surrogate model")
        plt.xlabel("Time")
        plt.plot(np.arange(0,len(ramification)),starting_point_prey,'-',color = colors[0])
        plt.plot(np.arange(len(ramification)-1,len(ramification)+forecasting_surrogate.shape[1]-1),np.vstack(tt).T.squeeze(),'-',color=colors[0],alpha=0.5)
        plt.ylim(0,500)
    for tt in diffusion_pred:
        plt.plot(np.arange(0,len(ramification)),starting_point_pred,'-',color = colors[1])
        plt.plot(np.arange(len(ramification)-1,len(ramification)+forecasting_surrogate.shape[1]-1),np.vstack(tt).T.squeeze(),'-',color=colors[1],alpha=0.5)
        plt.ylim(0,500)
    
    fig.savefig(image_path)
    plt.close()


def main():

    parser = argparse.ArgumentParser(description='Train surrogate model on an ABM')
    parser.add_argument('--model_type', type=str, default='surrogate', help='choose between surrogate and ablation (default: surrogate)')
    parser.add_argument('--psi', type=str, default='psi1', help='choose parameter between psi1, psi2, psi3, psi4 (default: psi1)')
    args = parser.parse_args()

    model_type = args.model_type
    psi = args.psi

    save_dir = f'./result/predatorprey/{model_type}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)   


    """INITIALIZING GROUND TRUTH GENERATOR"""

    featurizer = PredPreyFeaturizer()
    M = 2*GRID_SIZE*GRID_SIZE
    N_MAX = GRID_SIZE*GRID_SIZE
    grid_fill_p = 0.3
    predators_p = 0.5

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
    
    psi_dict = {'psi1':Psi_1,
                'psi2':Psi_2,
                'psi3':Psi_3,
                'psi4':Psi_4}


    
    # Get training ramification to start the simulation from the last state in training

    with gzip.open(f'./ramifications/predatorprey/ramification_training_{psi}.pickle.gz', 'rb') as f:
        ramification = pickle.load(f)
    initial_condition = ramification[-1][0]

    predatorprey_abm = PredatorPreyABM(Psi=psi_dict[psi],n_agents=M,max_agent=N_MAX,
                                    grid_fill_p=grid_fill_p,predators_p=predators_p,
                                    grid_size=GRID_SIZE)
    n_samples = 100
    simulation_time = 25
    ground_truth = []
    
    for t in tqdm(range(n_samples),desc='GT forecasting'):
        ground_truth.append(np.vstack(predatorprey_abm.generate_ramifications_from_initial_condition(
                                                    initial_condition=initial_condition,
                                                    n_timesteps=simulation_time,
                                                    n_ramifications=1,
                                                    seed=random.randint(0,10000))))
        
    ground_truth = np.array(ground_truth)
    path = os.path.join(save_dir, f"forecasting_ground_truth_{psi}.pickle")
    with open(path,'wb') as f:
        pickle.dump(ground_truth, f)

    

    surrogate_model = []
    if model_type == 'surrogate':
        # Learnig
        diffusion_timesteps = torch.load(f'./trained_models/predatorprey/{model_type}/model_{model_type}_{psi}.pth')['diffusion_timesteps']
        learning_rate = torch.load(f'./trained_models/predatorprey/{model_type}/model_{model_type}_{psi}.pth')['learning_rate']
        aggregation = torch.load(f'./trained_models/predatorprey/{model_type}/model_{model_type}_{psi}.pth')['aggregation']
        model = NNModel(n_features = 6, learning_rate=learning_rate, abm_featurizer = PredPreyFeaturizer(), diffusion_timesteps=diffusion_timesteps,aggregation=aggregation)
        model.ld_model.load_state_dict(torch.load(f'./trained_models/predatorprey/{model_type}/model_{model_type}_{psi}.pth')['ld_model_state_dict'])
        model.graph_model.load_state_dict(torch.load(f'./trained_models/predatorprey/{model_type}/model_{model_type}_{psi}.pth')['graph_model_state_dict'])
                                          
        for tt in tqdm(range(n_samples),desc='Surrogate forecast'):
            simulated_model = SimulatedABM(initial_state=initial_condition,model=model) 
            seed = random.randint(0,10000)
            surrogate_model.append(simulated_model.simulation_pp(simulation_time=simulation_time,seed=seed))

    elif model_type == 'ablation':
        diffusion_timesteps = torch.load(f'./trained_models/predatorprey/{model_type}/model_{model_type}_{psi}.pth')['diffusion_timesteps']
        learning_rate = torch.load(f'./trained_models/predatorprey/{model_type}/model_{model_type}_{psi}.pth')['learning_rate']
        model = NNModel_ablation(n_features = 6, learning_rate=learning_rate, abm_featurizer = PredPreyFeaturizer(), diffusion_timesteps=diffusion_timesteps,
                                        domain_dim=featurizer.scale_abm_state(ramification[0][0]).flatten().shape[0])
        model.ld_model.load_state_dict(torch.load(f'./trained_models/predatorprey/{model_type}/model_{model_type}_{psi}.pth')['ld_model_state_dict'])

        for tt in tqdm(range(n_samples),desc='Surrogate forecast'):
            simulated_model = SimulatedABM_ablation1(initial_state=initial_condition,model=model) 
            seed = random.randint(0,10000)
            surrogate_model.append(simulated_model.simulation_pp(simulation_time=simulation_time,seed=seed))
    
    surrogate_model = np.array(surrogate_model)
    path = os.path.join(save_dir, f"forecasting_{model_type}_{psi}.pickle")
    with open(path,'wb') as f:
        pickle.dump(surrogate_model, f)

    image_path = os.path.join(save_dir, f"forecasting_{model_type}_{psi}.pdf")
    plot_forecast(ramification=ramification,forecasting_GT=ground_truth,forecasting_surrogate=surrogate_model,image_path=image_path)
    
    aggregate_array = aggregate(ground_truth,surrogate_model)
    path = os.path.join(save_dir, f"smape_{model_type}_{psi}.pickle")
    with open(path,'wb') as f:
        pickle.dump(aggregate_array, f)
    

    # Load testing ramification to evaluate micro distributions

    with open(f'./ramifications/predatorprey/ramification_testing_{psi}.pickle','rb') as f:
        future_ramifications = pickle.load(f)

    micro_distributions = generate_distributions(future_ramifications,model)
    path = os.path.join(save_dir, f"distributions_{model_type}_{psi}.pickle")
    with open(path,'wb') as f:
        pickle.dump(micro_distributions, f)

    emd_array = micro_level_metric(future_ramifications,micro_distributions)
    

    path = os.path.join(save_dir, f"emd_{model_type}_{psi}.pickle")
    with open(path,'wb') as f:
        pickle.dump(emd_array, f)


if __name__ == "__main__":
    main()