from models.abm.abm import ABM, ABMFeaturizer
from models.abm.pp_functions import update_parental

import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm.auto import tqdm 
from tqdm import tqdm


# Utility functions 

def exists(x):
    return x is not None

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

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
    
    def __init__(self,dim_in,dim_out,*,condition_emb_dim=None,res_unit=False,norm=False):
        super().__init__()
        
        # Initialize condition block
        
        self.block1 = Block(dim_in, dim_out)
        self.block2 = Block(dim_out,dim_out)
        
        if exists(condition_emb_dim):
            self.condition_mlp = nn.Sequential(nn.LeakyReLU(negative_slope=0.1),nn.Linear(condition_emb_dim,dim_out))
        else:
            self.condition_mlp = None

        if res_unit:
            self.res_block = Block(dim_in,dim_out) if dim_in != dim_out else nn.Identity()
        else:
            self.res_block = self.res_block = None

        if norm:
            self.normalization = nn.LayerNorm(dim_in)
        else:
            self.normalization = None
        
        
    def forward(self,x,condition=None):

        # Layer normalization
        if exists(self.normalization):
            h = self.normalization(x) 

        h = self.block1(h)

        # Condition application
        if exists(self.condition_mlp) and exists(condition):
            condition_emb = self.condition_mlp(condition)
            h = h + condition_emb
    
        h = self.block2(h)

        # Residual connection
        if exists(self.res_block):
            h = h + self.res_block(x)           
        
        return h         

class Network(nn.Module):
    def __init__(self,
                 dim,
                 hid_dims=(2,3),
                 with_time_emb=True,
                 with_state_condition=True,
                 with_domain = True,
                 domain_dim = 16,
                ):
        super().__init__()
        
        self.init_dim = dim
        self.final_dim = dim
        self.condition_dim = 256
        
        if with_time_emb:  
            self.time_mlp = nn.Sequential(
                SinusoidalTimeEmbeddings(self.init_dim),
                nn.Linear(self.init_dim,self.condition_dim),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(self.condition_dim,self.condition_dim)
            )  
        else:
            time_dim = None
            self.time_mlp = None
            
        if with_state_condition:
            self.state_mlp = nn.Sequential(
                nn.Linear(self.init_dim,self.condition_dim),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(self.condition_dim,self.condition_dim),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(self.condition_dim,self.condition_dim)
                )
        else:
            self.state_mlp = None    
                   
        if with_domain:
            self.domain_mlp = nn.Sequential(
                nn.Linear(domain_dim,self.condition_dim),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(self.condition_dim,self.condition_dim),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(self.condition_dim,self.condition_dim),
            )
            
        # Initial block
            
        self.init_block = Block(self.init_dim,hid_dims[0])
        self.init_block.apply(init_weights)

        # Hidden layers with conditioning

        self.blocks = nn.ModuleList([])
        for (dim_in, dim_out) in list(zip(hid_dims[:-1],hid_dims[1:])):
            self.blocks.append(
                NetBlock(dim_in,dim_out,condition_emb_dim=self.condition_dim,norm=True,res_unit=True)
                )    
        self.blocks.apply(init_weights)

        # Final block

        self.final_block = Block(hid_dims[-1],self.final_dim)
        self.final_block.apply(init_weights)    
            
    def forward(self,x,state,domain,time):

        """ Forward pass of the network. """

        # Condition vector creation
        state = self.state_mlp(state) if exists(self.state_mlp) else None
        domain_emb = self.graph_mlp(domain) if exists(self.graph_mlp) else None
        tau_emb = self.time_mlp(time) if exists(self.time_mlp) else None      
        condition = state + domain_emb + tau_emb 
        
        # Initial block
        x = self.init_block(x)

        # Hidden layers
        for block in self.blocks:
            x = block(x,condition)

        # Final block
        x = self.final_block(x)

        return x

# Defining diffusion process

class Diffusion:

    def __init__(self,diffusion_timesteps=100,beta_start=0.0001,beta_end=0.02):

        """ Variance schedule and computing alphas """

        self.diffusion_timesteps = diffusion_timesteps
        x = torch.linspace(0,diffusion_timesteps,diffusion_timesteps+1)
        x = torch.cos(x/diffusion_timesteps * torch.pi)
        self.betas = beta_start + 0.5*(beta_end-beta_start)*(1-x)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        """ Calculations for diffusion process """

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    
    # Forward diffusion process

    def q_sample(self,x_start, tau, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_tau = extract(self.sqrt_alphas_cumprod, tau, x_start.shape)
        sqrt_one_minus_alphas_cumprod_tau = extract(
            self.sqrt_one_minus_alphas_cumprod, tau, x_start.shape
        )
        return sqrt_alphas_cumprod_tau * x_start + sqrt_one_minus_alphas_cumprod_tau * noise
    

# Defining the whole model + training + sampling

class NNModel_ablation:

    """ Our neural network model. """

    def __init__(self, n_features, learning_rate, abm_featurizer: ABMFeaturizer, diffusion_timesteps, domain_dim):
        
        self.abm_featurizer = abm_featurizer
        self.feature_dim = n_features
        self.dynamic_feature_idx = self.abm_featurizer.get_shape_state_features()[0]
        self.state_dim = self.feature_dim + self.abm_featurizer.get_shape_state_features()[0]

        self.ld_hidden_dims = [128,256,1024,1024,256,128]
        self.latent_dim = domain_dim
        self.lr_ld = learning_rate
        self.ld_model = Network(dim = self.feature_dim,
                hid_dims=self.ld_hidden_dims,
                with_time_emb=True,
                with_domain=True,
                domain_dim=self.latent_dim,
                with_state_condition=True
        )
        self.diffusion_timesteps = diffusion_timesteps
        self.optimizer = Adam(self.ld_model.parameters(),lr=self.lr_ld)
        self.losses = []
        

    def p_losses(self,x_start,state,label,tau,tau_max,noise=None,loss_type="l2"):
        if noise is None:
            noise = torch.randn_like(x_start)
        diffusion = Diffusion(diffusion_timesteps=tau_max)
        x_noisy = diffusion.q_sample(x_start=x_start, tau=tau, noise=noise)
        predicted_noise = self.ld_model(x_noisy, state, label, tau)

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


        """ Train the model using a `ramification` dataset """

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ld_model.to(device)

        # Get the number of timesteps and runs from ramification dataset
        n_timesteps = len(ramifications[1:])
        n_runs = len(ramifications[1])

        # Randomize runs and timesteps
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

                self.optimizer.zero_grad()

                t, run = sample_number[example]
                
                prev_state = ramifications[t][0]
                # Scale features for the current state
                state_t0 = self.abm_featurizer.scale_abm_state(prev_state)

                # Get domain 
                n_agents = prev_state.shape[0]
                domain = np.broadcast_to(state_t0.flatten(),shape=(n_agents,state_t0.flatten().shape[0]))

                next_state = ramifications[t+1][run]
                # Scale features for future state
                state_t1 = self.abm_featurizer.scale_abm_state(next_state)
                    
                state_t1 = torch.tensor(state_t1[:,self.dynamic_feature_idx:]).to(torch.float32).to(device)
                t_diffusion = torch.randint(0, self.diffusion_timesteps, (n_agents,), device = device).long()
                    
                #calculate hidden rappresentation relative to timestep t 
                prev_state_condition = torch.tensor(state_t0[:,self.dynamic_feature_idx:]).to(torch.float32).to(device)
                domain_condition = torch.tensor(domain).to(torch.float32).to(device)
                                        
                    #calculate loss
                loss = self.p_losses(state_t1, state = prev_state_condition, label = domain_condition ,tau = t_diffusion, tau_max = self.diffusion_timesteps, loss_type="l2")
                self.losses.append(loss.item())

                    #propagate gradient back to diffusion model and GNN
                loss.backward()
                self.optimizer.step()


    @torch.no_grad()
    def p_sample(self,x,t,t_index,state,label):
        diffusion = Diffusion(diffusion_timesteps=self.diffusion_timesteps)
        betas_t = extract(diffusion.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            diffusion.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(diffusion.sqrt_recip_alphas, t, x.shape)

        # Generation algorithm

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.ld_model(x,state,label,t) / sqrt_one_minus_alphas_cumprod_t
        )
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(diffusion.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise     
    @torch.no_grad()
    def next_step_samples(self, state: np.ndarray, seed=42, n_samples=1) -> np.ndarray:
        """ Given a matrix representing the state at t, returns a list of possible outcomes,
            where each element of the list is a matrix representing the state at t+1.
        """
        device = next(self.ld_model.parameters()).device
        b = n_samples

        state_array = self.abm_featurizer.scale_abm_state(state)
        n_agents = state.shape[0]
        domain = np.broadcast_to(state_array.flatten(),shape=(n_agents,state_array.flatten().shape[0]))
        label = torch.tensor(domain).to(torch.float32).to(device)
        condition = torch.tensor(state_array).to(torch.float32).to(device)
        state_tensor = condition[:,self.dynamic_feature_idx:].to(torch.float32).to(device)
       
        n, c = state_tensor.size()
        samples = []

        torch.manual_seed(seed=seed)
        for j in range(b):
            # Start from random sample
            sample = torch.randn((n,c), device=device,)
            # Learned reverse diffusion process
            for i in reversed(range(0,self.diffusion_timesteps)):
                sample = self.p_sample(sample, torch.full((n,), i, device=device, dtype=torch.long), i, state = state_tensor,label = label)
            samples.append(sample.cpu().numpy())
        samples = np.array(samples)
        return self.abm_featurizer.unscale_abm_state(samples)


class SimulatedABM_ablation1(ABM):
    """ Class that takes a trained model and puts into the same interface used by ABM.
    """
    def __init__(self, initial_state: np.ndarray, model: NNModel_ablation):
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
            state_variables = self.model.next_step_samples(state=new_state,seed=seed,n_samples=1)[0]
            new_state = np.hstack((state_variables,colors))
            simulation_array[t] = new_state
        
        return simulation_array
    
    def simulation_pp(self,simulation_time: int, seed: int) -> np.ndarray:
        self.initial_state_np = np.delete(self.initial_state_np,7,axis=1)
        n, c = np.shape(self.initial_state_np)
        simulation_array = np.empty((simulation_time,n,c))
        kind = self.initial_state_np[:,0]
        kind_expanded = np.expand_dims(self.initial_state_np[:,0],axis=0).T 
        family = self.initial_state_np[:,-2:]
        new_state = self.initial_state_np
        simulation_array[0] = new_state
        new_seed = seed
        for t in tqdm(range(1,simulation_time),desc='Simulation'): 
            state_variables = self.model.next_step_samples(state=new_state,seed=new_seed,n_samples=1)[0]
            family = update_parental(torch.tensor(family),torch.tensor(kind),torch.tensor(state_variables[:,:4].T)).numpy()
            new_state = np.hstack((kind_expanded,state_variables,family))
            simulation_array[t] = new_state
            new_seed = random.randint(0,10000)
        
        return simulation_array
