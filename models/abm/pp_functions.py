import time
import torch
import numpy as np

### HELPER FUNCTIONS ###
def generate_action_set() -> torch.Tensor:
    """
    Helper function that simply contains the initialization of the actions' set
    """
    die = torch.tensor([1,0,0,0,0,0], dtype=torch.int8)
    move = torch.tensor([0,1,0,0,0,0], dtype=torch.int8)
    get_pregnant = torch.tensor([0,0,1,0,0,0], dtype=torch.int8)
    back_to_alive = torch.tensor([0,0,0,1,0,0], dtype=torch.int8)
    same_dead = torch.tensor([0,0,0,0,1,0])
    same_not_born = torch.tensor([0,0,0,0,0,1], dtype=torch.int8)

    return torch.vstack((die, move, get_pregnant, back_to_alive, same_dead, same_not_born))


def number_initialized_agents(grid_fill_p, grid_size):
    """How many agents populate the grid given the percentage of grid occupied and its dimension"""
    N = torch.tensor(grid_size*grid_size*grid_fill_p).round().type(torch.int64)
    return N


def number_initialized_predators(total_agents, predators_p): 
    """How many of the initialized agents are predators given their percentage over the total number of agents"""
    N_pred = torch.tensor(total_agents*predators_p).round().type(torch.int64)
    return N_pred


def init_parental(K,S):

    pred_eligible_parent = torch.where(torch.logical_and(K == 1, S[0,:] == 1))[0]
    prey_eligible_parent = torch.where(torch.logical_and(K == 0, S[0,:] == 1))[0]

    pred_not_born_yet = torch.where(torch.logical_and(K == 1, S[3,:]==1))[0]
    prey_not_born_yet = torch.where(torch.logical_and(K == 0, S[3,:]==1))[0]

    pred_eligible_newborns = pred_not_born_yet[:pred_eligible_parent.size()[0]]
    prey_eligible_newborns = prey_not_born_yet[:prey_eligible_parent.size()[0]]

    parental = torch.full((S.size()[-1],2),fill_value=-1)
    parental[pred_eligible_parent,0] = pred_eligible_parent
    parental[prey_eligible_parent,0] = prey_eligible_parent

    if pred_eligible_newborns.size()[0] < pred_eligible_parent.size()[0]:
        parental[pred_eligible_parent[:pred_eligible_newborns.size()[0]],1] = pred_eligible_newborns
    else:
        parental[pred_eligible_parent,1] = pred_eligible_newborns
    if prey_eligible_newborns.size()[0] < prey_eligible_parent.size()[0]:
        parental[prey_eligible_parent[:prey_eligible_newborns.size()[0]],1] = prey_eligible_newborns
    else:
        parental[prey_eligible_parent,1] = prey_eligible_newborns

    parental[pred_eligible_newborns,1] = pred_eligible_parent[:pred_eligible_newborns.size()[0]]
    parental[prey_eligible_newborns,1] = prey_eligible_parent[:prey_eligible_newborns.size()[0]]

    return parental

def update_parental(parental_original,K,S):
    """
    Function takes parental matrix and first checks if parents have died
    or eligible newborns have come alive, then it updates the parental
    matrix to be fed to determine_newborns function
    """

    parental = parental_original.clone()
       
    pred_parents = torch.logical_and(K == 1, torch.logical_and(parental[:,0] > -1, torch.logical_not(torch.isnan(parental[:,0]))))
    prey_parents = torch.logical_and(K == 0, torch.logical_and(parental[:,0] > -1, torch.logical_not(torch.isnan(parental[:,0]))))
  
    # Eliminate dead parents and take aside orphans

    dead_pred_parents = torch.logical_and(pred_parents, S[1,:] == 1)
    pred_orphans = parental[dead_pred_parents,1].long()
    parental[dead_pred_parents,0] = torch.nan
    parental[dead_pred_parents,1] = torch.nan

    dead_prey_parents = torch.logical_and(prey_parents, S[1,:] == 1)
    prey_orphans = parental[dead_prey_parents,1].long()
    parental[dead_prey_parents,0] = torch.nan
    parental[dead_prey_parents,1] = torch.nan

    pred_parents = torch.logical_and(K == 1, torch.logical_and(parental[:,0] > -1, torch.logical_not(torch.isnan(parental[:,0]))))
    prey_parents = torch.logical_and(K == 0, torch.logical_and(parental[:,0] > -1, torch.logical_not(torch.isnan(parental[:,0]))))

    pred_pc = parental[pred_parents].long()
    prey_pc = parental[prey_parents].long()

    # Take aside newly born who can be new parents

    pred_newparents = (S[0,pred_pc[:,1]] == 1)
    pred_parent_child = pred_pc[pred_newparents]
    parental[pred_parent_child[:,1],0] = pred_parent_child[:,1].float()
    parental[pred_parent_child[:,1],1] = -1
    parental[pred_parent_child[:,0],1] = -1

    prey_newparents = (S[0,prey_pc[:,1]] == 1)
    prey_parent_child = prey_pc[prey_newparents]
    parental[prey_parent_child[:,1],0] = prey_parent_child[:,1].float()
    parental[prey_parent_child[:,1],1] = -1
    parental[prey_parent_child[:,0],1] = -1

    eligible_parents = torch.logical_and(parental[:,0] > -1, parental[:,1] == -1)
    eligible_newborns = torch.logical_and(parental[:,0] == -1, parental[:,1] == -1)


    pred_eligible_newborn = torch.where(torch.logical_and(K == 1, eligible_newborns))[0]
    prey_eligible_newborn = torch.where(torch.logical_and(K == 0, eligible_newborns))[0]

    pred_eligible_parents = torch.where(torch.logical_and(K == 1, eligible_parents))[0]
    prey_eligible_parents = torch.where(torch.logical_and(K == 0, eligible_parents))[0]
    
    # Put all together and also add possible new borns 

    if pred_eligible_parents.size()[0] >= pred_orphans.size()[0]:
        delta_pred = pred_eligible_parents.size()[0] - pred_orphans.size()[0]
        parental[pred_eligible_parents[:pred_orphans.size()[0]],1] = pred_orphans.float()
        parental[pred_orphans,1] = pred_eligible_parents[:pred_orphans.size()[0]].float()
        max_available_pred = pred_eligible_newborn[:delta_pred].size()[0]
        parental[pred_eligible_parents[pred_orphans.size()[0]:pred_orphans.size()[0]+max_available_pred],1] = pred_eligible_newborn[:delta_pred].float()
        parental[pred_eligible_newborn[:delta_pred],1] = pred_eligible_parents[pred_orphans.size()[0]:pred_orphans.size()[0]+max_available_pred].float()
    else:
        parental[pred_eligible_parents[:pred_orphans.size()[0]],1] = pred_orphans[:pred_eligible_parents.size()[0]].float()
        parental[pred_orphans[:pred_eligible_parents.size()[0]],1] = pred_eligible_parents[:pred_orphans.size()[0]].float()
        parental[pred_orphans[pred_eligible_parents.size()[0]:].long(),0] = -1
        parental[pred_orphans[pred_eligible_parents.size()[0]:].long(),1] = -1

    if prey_eligible_parents.size()[0] >= prey_orphans.size()[0]:
        delta_prey = prey_eligible_parents.size()[0] - prey_orphans.size()[0]
        parental[prey_eligible_parents[:prey_orphans.size()[0]],1] = prey_orphans.float()
        parental[prey_orphans,1] = prey_eligible_parents[:prey_orphans.size()[0]].float()
        max_available_prey = prey_eligible_newborn[:delta_prey].size()[0]
        parental[prey_eligible_parents[prey_orphans.size()[0]:prey_orphans.size()[0]+max_available_prey],1] = prey_eligible_newborn[:delta_prey].float()
        parental[prey_eligible_newborn[:delta_prey],1] = prey_eligible_parents[prey_orphans.size()[0]:prey_orphans.size()[0]+max_available_prey].float()
    else:
        parental[prey_eligible_parents[:prey_orphans.size()[0]],1] = prey_orphans[:prey_eligible_parents.size()[0]].float()
        parental[prey_orphans[:prey_eligible_parents.size()[0]],1] = prey_eligible_parents[:prey_orphans.size()[0]].float()
        parental[prey_orphans[prey_eligible_parents.size()[0]:].long(),0] = -1
        parental[prey_orphans[prey_eligible_parents.size()[0]:].long(),1] = -1

    return parental


def determine_newborns(K, S_current, A_prev, parental):
    """Function that evaluates which agents are giving birth (parents) 
    and which newborn agents must be initialized.
    Returns masks for parents and children, for each kind

    Parameters:
    ----------
    K : torch.Tensor
        Kind array
    S_current : torch.Tensor
        State tensor at the current timestep (that is, it was already updated)
    A_prev : torch.Tensor
        Action tensor at the previous timestep

    Returns:
    x, X, y, Y : torch.Tensor (Boolean)
        Boolean masks: x are predators, y are preys, capital are parents
    """

    pred_parents = torch.where(torch.logical_and(K == 1, A_prev[3,:]==1))[0]
    prey_parents = torch.where(torch.logical_and(K == 0, A_prev[3,:]==1))[0]

    pred_newborns = parental[pred_parents,1].long()
    prey_newborns = parental[prey_parents,1].long()

    return pred_newborns, prey_newborns, pred_parents, prey_parents, parental


def uniform_move_(mask, pos, set_adj, grid_size = 32):
    """Function that updates the position X  of moving agents according to uniform selection
    over the neighboring nodes of the network.
    In_place function.
    
    Args:
        mask: boolean array of agents whose movement must be updated
        pos: array of positions of the agents in the form (x,y)
        set_adj: array of delta steps to be taken in order to reach whatever is considered 
            to be an adjacent node
        grid_size: size of the grid in which agents are moving. Necessary to correctly evaluate 
            boundary conditions
    
    Return:
        new_pos: updated position array
    """

    pos[mask,:] += set_adj[torch.randint(0, set_adj.size(0), (mask.size()[0],))] 
    return pos % grid_size

################# VARIABLES EVOLUTIONS ########################
def evolve_S(A_prev):
    """Evolve State tensor using given performed Actions"""
    U = torch.tensor([[0, 1, 0, 1, 0, 0],
                      [1, 0, 0, 0, 1, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1]])
    return torch.matmul(U, A_prev)


def update_S_newborns(S, K, A_prev,parental):
    """Update the State of the newborns agents to Alive, after their initialization happened during the Evolve_X() step"""
    
    pred_newborns, prey_newborns, _, _, _ = determine_newborns(K,S,A_prev,parental)

    newborns_mask = torch.concatenate((pred_newborns,prey_newborns)).long()
    S[:, newborns_mask] = torch.tensor([1,0,0,0], dtype=torch.int64).unsqueeze(1)
    
    return S


def evolve_X(K, S_current, X_prev, A_prev, parental, grid_size, adj_set, adj_parent_set):
    """Evolve Position tensor X"""
    
    X = torch.clone(X_prev)
    
    # Deal with Newborns first 
    
    pred_newborns, prey_newborns, pred_parents, prey_parents, _ = determine_newborns(K, S_current, A_prev, parental)
    newborns_mask = torch.concatenate((pred_newborns,prey_newborns)).long()
    
    X[pred_newborns] = X_prev[pred_parents]
    X[prey_newborns] = X_prev[prey_parents]

    # Newborns can be born in adjacent nodes as well
    X = uniform_move_(newborns_mask, X, adj_parent_set, grid_size)
    
    #Die
    die_mask = (A_prev[0,:]==1)
    X[die_mask, :] = torch.nan
    #Move
    move_mask = torch.where(A_prev[1,:]==1)[0]
    X = uniform_move_(move_mask, X, adj_set, grid_size)
    return X


def evolve_NO(K, X, grid_size):
    """Evaluate the number of agents of opposite kind nearby each agent
    """
    delta_matrix =  torch.abs(X[:, None] - X)
    effective_delta_matrix = torch.minimum(delta_matrix, grid_size - delta_matrix)
    #Sum distance over x-coord + y-coord
    distance_matrix = effective_delta_matrix.sum(axis = 2) 
    to_sum = torch.logical_and(
                            torch.logical_xor(K[:,None], K), 
                            distance_matrix <= 1,       
                            )
    NO = torch.sum(to_sum, axis=1)
    return NO


def evolve_A(Z, K, S, NO, A_set):
    """Evolve the Action tensor using the current step, system's state information"""

    pred_alive_prey_mask      = torch.all(torch.vstack((S[0, :]==1, K==1, NO>=1)), axis=0)
    pred_alive_noprey_mask    = torch.all(torch.vstack((S[0, :]==1, K==1, NO==0)), axis=0)
    prey_alive_predator_mask  = torch.all(torch.vstack((S[0, :]==1, K==0, NO>=1)), axis=0)
    prey_alive_nopredator_mask= torch.all(torch.vstack((S[0, :]==1, K==0, NO==0)), axis=0)
    pregnant_mask = (S[2, :]==1)
    same_dead_mask = (S[1, :]==1)
    not_born_yet_mask = (S[3, :]==1)


    mask = torch.vstack((pred_alive_prey_mask,      
                        pred_alive_noprey_mask,    
                        prey_alive_predator_mask,  
                        prey_alive_nopredator_mask,
                        pregnant_mask,
                        same_dead_mask,
                        not_born_yet_mask
                        ))         
    z_idx = torch.nonzero(mask.T)[:,1]
    A = A_set[torch.multinomial(Z[z_idx], 1, replacement = True), :].T.squeeze()
    return A


########################### ABM ############################################

# ABM Initialization
def init_abm(Z, M, N, N_pred, grid_size, A_set):
    """Initializes Predator-Prey ABM
    
    Parameters
    ----------
    Z : torch.Tensor
        Parameters' matrix
    M : int
        Total number of agents in the simulation (included the one not initialized yet)
    N : int
        Number of agents to initialize
    N_pred : int
        Number of predators to initialize
    grid_size : int
        Dimension of the square grid
    A_set : torch.Tensor
        Tensor that encodes one-hot the possible actions
    """
    # Kind matrix
    K = torch.cat((torch.ones(N_pred), torch.zeros(N-N_pred), torch.randint(0,2, (M-N,)))).type(torch.int8)
    
    # State matrix
    S = torch.zeros((4, M), dtype = torch.int8)
    S[0, :N] = 1
    S[3, N:M] = 1

    # Position matrix
    X = torch.full((M, 2), torch.nan)
    pos_idx_rng = torch.randperm(grid_size * grid_size)[:N]
    X[:N, 0] = pos_idx_rng % grid_size  # x coordinates
    X[:N, 1] = pos_idx_rng // grid_size  # y coordinates

    # Number of Opposite agents (NO)
    NO = torch.empty((M))
    NO = evolve_NO(K, X, grid_size)
    
    # Action Matrix
    A = torch.zeros((6, M), dtype=torch.int8)
    A = evolve_A(Z, K, S, NO, A_set)

    parental = init_parental(K,S)

    return K, S, X, NO, A, parental

def evolve_one_step(K, X_prev, A_prev, parental, Z, grid_size, A_set, adj_set, adj_parent_set):
    """Evolve ABM variables one step
    
    Parameters:
    -----------
    grid_size : int 
        Dimension of the square grid
    A_set : torch.Tensor
        Tensor that encodes one-hot the possible actions
    adj_set : torch.Tensor
        Set of deltas to identify neighbouring nodes
    adj_parent_set
        Set of deltas to identify neighbouring + current node
    """
    
    S = evolve_S(A_prev)
    X = evolve_X(K,S ,X_prev, A_prev, parental, grid_size, adj_set, adj_parent_set)
    S = update_S_newborns(S, K, A_prev,parental)
    NO = evolve_NO(K, X, grid_size)
    A = evolve_A(Z, K, S, NO, A_set)
    return S, X, NO, A

