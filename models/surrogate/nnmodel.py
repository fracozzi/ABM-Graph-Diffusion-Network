from models.abm.abm import ABM, ABMFeaturizer
from models.abm.pp_functions import update_parental

import numpy as np
import math
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from tqdm.auto import tqdm 
from einops import rearrange
import seaborn as sns
import torch_geometric as tg
from torch_geometric.nn import MessagePassing
from tqdm import tqdm


# Utility functions 

def exists(x):
    return x is not None

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.01)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# Model components definitions

class SinusoidalTimeEmbeddings(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        
        half_dim = self.dim // 2
        if half_dim == 0:
            half_dim = self.dim
           
        embeddings = np.log(10000) / half_dim
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), - embeddings.cos()), dim=-1)
        
        return embeddings
    
class Block(nn.Module):
    
    def __init__(self,dim_in,dim_out):
        super().__init__()
        self.lin = nn.Linear(dim_in,dim_out)
        self.act = nn.LeakyReLU(negative_slope=0.1)

    def forward(self,x):               
        x = self.lin(x)
        x = self.act(x)
        return x
    
class NetBlock(nn.Module):
    
    def __init__(self,dim_in,dim_out,*,time_emb_dim=None,state_dim=None,res_unit=False,norm=False):
        super().__init__()
        
        self.block1 = Block(dim_in, dim_out)
        self.block2 = Block(dim_out,dim_out)
        
        if exists(time_emb_dim):
            self.time_mlp = nn.Sequential(nn.LeakyReLU(negative_slope=0.1),nn.Linear(time_emb_dim,dim_out))
        else:
            None
        if exists(state_dim):
            self.state_mlp = nn.Sequential(nn.LeakyReLU(negative_slope=0.1),nn.Linear(state_dim,dim_out))
        else:
            self.state_mlp = None
        if res_unit:
            self.res_block = Block(dim_in,dim_out) if dim_in != dim_out else nn.Identity()
        else:
            self.res_block = self.res_block = None
        if norm:
            #self.normalization = nn.GroupNorm(4,dim_in)
            #self.normalization = nn.BatchNorm1d(dim_in)
            self.normalization = nn.LayerNorm(dim_in)
        else:
            self.normalization = None
        
        
    def forward(self,x,time=None,state=None):
        if exists(self.normalization):
            h = self.normalization(x) 
        h = self.block1(h)
        if exists(self.time_mlp) and exists(time):
            time_emb = self.time_mlp(time)
            h = h + time_emb
        if exists(self.state_mlp) and exists(state):
            state_emb = self.state_mlp(state)
            h = h + state_emb   

        h = self.block2(h)
        if exists(self.res_block):
            h = h + self.res_block(x)           
        
        return h      

class Network(nn.Module):
    def __init__(self,
                 dim,
                 hid_dims=(2,3),
                 with_time_emb=True,
                 with_graph = True,
                 graph_dim = 16,
                 with_state_condition=True,
                ):
        super().__init__()
        
        self.init_dim = dim
        self.final_dim = dim
        
        if with_time_emb:  
            time_dim = 256
            self.time_mlp = nn.Sequential(
                SinusoidalTimeEmbeddings(self.init_dim),
                nn.Linear(self.init_dim,time_dim),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(time_dim,time_dim)
            )  
        else:
            time_dim = None
            self.time_mlp = None
            
        if with_state_condition:
            state_dim = 256
            self.state_mlp = nn.Sequential(
                nn.Linear(self.init_dim,state_dim),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(state_dim,state_dim),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(state_dim,time_dim)
                )
        else:
            state_dim = None
            self.state_mlp = None    
                   
        if with_graph:
            condition_dim = 256
            self.graph_mlp = nn.Sequential(
                nn.Linear(graph_dim,condition_dim),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(condition_dim,condition_dim),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(condition_dim,time_dim),
            )
            
        self.init_block = Block(self.init_dim,hid_dims[0])
        self.init_block.apply(init_weights)
        self.blocks = nn.ModuleList([])
        for (dim_in, dim_out) in list(zip(hid_dims[:-1],hid_dims[1:])):
            self.blocks.append(
                NetBlock(dim_in,dim_out,time_emb_dim=time_dim,state_dim=None,norm=True,res_unit=True)
                )    
        self.blocks.apply(init_weights)
        self.final_block = Block(hid_dims[-1],self.final_dim)
        self.final_block.apply(init_weights)   
            
    def forward(self,x,state,graph_condition,time):
        t_diffusion = self.time_mlp(time)
        graph = self.graph_mlp(graph_condition)
        if exists(state) and exists(self.state_mlp):
            state = self.state_mlp(state)
            t = t_diffusion + graph + state
        else:
            t = t_diffusion + graph
    
        x = self.init_block(x)
        for block in self.blocks:
            x = block(x,t)
        x = self.final_block(x)
        return x

class GNN(MessagePassing):
    def __init__(self,aggregation, in_channels, out_channels, hidden=[8,16,32]):
        super().__init__(aggr=aggregation)
        
        self.dims = list(zip(
                    [2 * in_channels] + hidden, # 2 is one local one neighbors
                    hidden + [out_channels])
                        )

        self.layers = nn.ModuleList([])
        for dim_in, dim_out in self.dims:
            self.layers.append(
                nn.Sequential(
                nn.Linear(dim_in,dim_out),
                nn.LeakyReLU(negative_slope=0.1)
                )
            )
               
    def forward(self,x,edge_index):

        messages = self.propagate(edge_index, x=x)
        h = torch.hstack([messages,x])

        for block in self.layers:
            h = block(h)
        return h
    
    def message(self, x_j):
        return x_j


# Defining diffusion process

class Diffusion:

    def __init__(self,diffusion_timesteps=100,beta_start=0.0001,beta_end=0.02):
        self.diffusion_timesteps = diffusion_timesteps
        x = torch.linspace(0,diffusion_timesteps,diffusion_timesteps+1)
        x = torch.cos(x/diffusion_timesteps * torch.pi)
        self.betas = beta_start + 0.5*(beta_end-beta_start)*(1-x)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    
    # forward diffusion
    def q_sample(self,x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# Defining the whole model + training + sampling

class NNModel:
    """ Our neural network model. """

    def __init__(self, n_features, learning_rate, abm_featurizer: ABMFeaturizer, diffusion_timesteps, aggregation):
        self.abm_featurizer = abm_featurizer
        #self.state_dim = self.abm_featurizer.get_shape_state_features[0]
        #self.feature_dim = self.abm_featurizer.get_shape_state_features[1]
        
        self.feature_dim = n_features
        self.state_dim = self.feature_dim + 2
        self.ld_hidden_dims = [128,256,1024,1024,256,128]
        # old gnn dims = [16,32,64]
        self.gnn_hidden_dims = [32,64,128]
        # old gnn latent dim = 128
        self.latent_dim = 256
        #self.lr_ld = 0.00005
        #self.lr_gnn = 0.00004
        self.lr_ld = learning_rate
        self.lr_gnn = 2*learning_rate
        self.diffusion_timesteps = diffusion_timesteps
        self.aggregation = aggregation
        
        self.graph_model = GNN(self.aggregation,self.state_dim,self.latent_dim,hidden=self.gnn_hidden_dims)
        self.ld_model = Network(dim = self.feature_dim,
                hid_dims=self.ld_hidden_dims,
                with_time_emb=True,
                with_graph=True,
                graph_dim=self.latent_dim,
                with_state_condition=True
        )
        self.optimizer1 = Adam(self.ld_model.parameters(),lr=self.lr_ld)
        self.optimizer2 = Adam(self.graph_model.parameters(),lr=self.lr_gnn)
        self.losses = []
        

    def p_losses(self,x_start,state,label,t,T,noise=None,loss_type="l2"):
        if noise is None:
            noise = torch.randn_like(x_start)
        diffusion = Diffusion(diffusion_timesteps=T)
        x_noisy = diffusion.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.ld_model(x_noisy, state, label, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    def train(self, ramifications, n_epochs=10):
        """ Train the model using a `ramifications` data set as generated by `generate_ramifications`. """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ld_model.to(device)
        self.graph_model.to(device)
        n_timesteps = len(ramifications[1:])
        n_runs = len(ramifications[1])
        runs = np.expand_dims(np.arange(0,n_runs),axis=-1)
        sample_number = np.empty((n_timesteps,n_runs,2))
        for t in range(n_timesteps):
            times = np.full((n_runs,1),t)
            sample_number[t] = np.hstack((times,runs))
        sample_number = np.vstack((sample_number))
        sample_number = sample_number.astype(int)
        for epoch in tqdm(range(n_epochs), desc="Training"):

            examples = random.sample(range(len(sample_number)),len(sample_number))
            
            for example in examples:

                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()

                t, run = sample_number[example]
                
                prev_state = ramifications[t][0]
                edges_t0 = torch.tensor(self.abm_featurizer.get_interaction_graph(prev_state)).to(device)
                state_t0 = self.abm_featurizer.scale_abm_state(prev_state)
                state_t0 = torch.tensor(state_t0).to(torch.float32).to(device)
                n_agents = prev_state.shape[0]

                next_state = ramifications[t+1][run]
                state_t1 = self.abm_featurizer.scale_abm_state(next_state)
                    
                state_t1 = torch.tensor(state_t1[:,2:]).to(torch.float32).to(device)
                t_diffusion = torch.randint(0, self.diffusion_timesteps, (n_agents,), device = device).long()
                    
                    #calculate hidden rappresentation relative to timestep t 
                graph_condition = self.graph_model(state_t0, edges_t0)
                prev_state_condition = state_t0[:,2:]
                                        
                    #calculate loss
                loss = self.p_losses(state_t1, state = prev_state_condition, label = graph_condition ,t = t_diffusion, T = self.diffusion_timesteps, loss_type="l2")
                self.losses.append(loss.item())

                    #propagate gradient back to diffusion model and GNN
                loss.backward()
                self.optimizer1.step()
                self.optimizer2.step()


    @torch.no_grad()
    def p_sample(self,x,t,t_index,state,label):
        diffusion = Diffusion(diffusion_timesteps=self.diffusion_timesteps)
        betas_t = extract(diffusion.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            diffusion.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(diffusion.sqrt_recip_alphas, t, x.shape)
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.ld_model(x,state,label,t) / sqrt_one_minus_alphas_cumprod_t
        )
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(diffusion.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise        
    
    @torch.no_grad()
    def next_step_samples(self, state: np.ndarray, seed=42, n_samples=1) -> np.ndarray:
        """ Given a matrix representing the state at t, returns a list of possible outcomes,
            where each element of the list is a matrix representing the state at t+1.
        """
        device = next(self.ld_model.parameters()).device
        b = n_samples

        edges = torch.tensor(self.abm_featurizer.get_interaction_graph(state)).to(device)
        state_array = self.abm_featurizer.scale_abm_state(state)
        condition = torch.tensor(state_array).to(torch.float32).to(device)
        state_tensor = condition[:,2:].to(torch.float32).to(device)
       
        n, c = state_tensor.size()
        #state_tensor_expand = state_tensor.unsqueeze(axis=0).expand((b,n,c)).to(torch.float32)
        label = self.graph_model(condition,edges)
        # start from pure noise (for each example in the batch)
        #torch.manual_seed(seed=seed)
        #img = torch.randn((n,c), device=device)
        imgs = []
        
        #for i in tqdm(reversed(range(0, diffusion_timesteps)), desc='sampling loop time step', total=diffusion_timesteps):
        for j in range(b):
            img = torch.randn((n,c), device=device,)
            for i in reversed(range(0,self.diffusion_timesteps)):
                img = self.p_sample(img, torch.full((n,), i, device=device, dtype=torch.long), i, state = state_tensor,label = label)
            imgs.append(img.cpu().numpy())
        imgs = np.array(imgs)
        return self.abm_featurizer.unscale_abm_state(imgs)
        #pass


class SimulatedABM(ABM):
    """ Class that takes a trained model and puts into the same interface used by ABM.
    """
    def __init__(self, initial_state: np.ndarray, model: NNModel):
        self.model = model
        self.initial_state_np = initial_state

    def initial_state(self):
        return self.initial_state_np

    def next_step(self, state: np.ndarray, seed: int) -> np.ndarray:
        """ Given a matrix representing the state at t, 
            returns a matrix representing state at t+1.
        """
        return self.model.next_step_samples(state=state, seed=seed, n_samples=1)[0]
    
    def simulation_schelling(self,simulation_time: int, seed: int) -> np.ndarray:
        n, c = np.shape(self.initial_state_np)
        simulation_array = np.empty((simulation_time,n,c))
        colors = np.expand_dims(self.initial_state_np[:,-1],axis=0).T 
        new_state = self.initial_state_np
        simulation_array[0] = new_state
        for t in range(1,simulation_time):
            seed = random.randint(0,10000)
            state_variables = self.model.next_step_samples(state=new_state,seed=seed,n_samples=1)[0]
            new_state = np.hstack((state_variables,colors))
            simulation_array[t] = new_state
        
        return simulation_array
    
    def simulation_pp(self,simulation_time: int, seed: int) -> np.ndarray:
        self.initial_state_np = np.delete(self.initial_state_np,7,axis=1)
        n, c = np.shape(self.initial_state_np)
        simulation_array = np.zeros((simulation_time,n,c))
        kind = self.initial_state_np[:,0]
        kind_expanded = np.expand_dims(self.initial_state_np[:,0],axis=0).T 
        family = self.initial_state_np[:,-2:]
        new_state = self.initial_state_np
        #new_state[:,-4:-2][np.isnan(new_state[:,-4:-2])] = -1 # Changed after new scaling
        simulation_array[0] = new_state
        new_seed = seed
        #for t in tqdm(range(1,simulation_time),desc='Simulation'): 
        for t in range(1,simulation_time): 
            state_variables = self.model.next_step_samples(state=new_state,seed=new_seed,n_samples=1)[0]
            family = update_parental(torch.tensor(family),torch.tensor(kind),torch.tensor(state_variables[:,:4].T)).numpy()
            new_state = np.hstack((kind_expanded,state_variables,family))
            simulation_array[t] = new_state
            new_seed = random.randint(0,10000)

        return simulation_array
    
    