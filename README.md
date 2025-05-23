# ABM-Graph-Diffusion-Network

This is the repository for the implementation of the models and the experiments described in the paper "Learning Individual Behavior in Agent-Based Models with Graph Diffusion Networks".

Graph Diffusion Network (GDN) is a framework that combines a graph neural network with a diffusion model to learn a differentiable _surrogate_ of an ABM, from ABM-generated data.
In this framework, a graph neural network captures the interactions that govern the evolution of each agent's state in response to other agents, while the diffusion model learns the distribution of possible state transitions, conditioned on these interactions.

## Installation

To get started with the GDN framework, start by cloning the repository:

   ```bash
   git clone https://github.com/username/gdn.git
   cd gdn
   ```

## Python Requirements

The following Python libraries are required to run GERN:

- **[PyTorch Geometric (pyg)](https://pytorch-geometric.readthedocs.io/):** Used for graph neural network implementation. It requires PyTorch, which will be installed as a dependency.
- **Jupyter:** To run and interact with notebooks.
- **Matplotlib:** For creating plots and visualizations.
- **NumPy, Pandas, SciPy, Scikit-learn**
- **TQDM:** For displaying progress bars when running experiments.

To install all the required Python libraries, use the `requirements.txt` file provided in the repository:

```bash
pip install -r requirements.txt
```

### Running Experiments

1. Build the ABM Datasets

Before training or evaluation, you need to generate the ABM datasets for the Schelling and Predator–Prey models.  From the repository root, run:
```bash
python models/abm/schelling.py
python models/abm/predatorprey.py
```
These scripts will produce the simulation trajectories and construct the graph‐structured datasets used by the GDN.

2.	Train the Surrogate Model

Once the datasets are ready, you can train the Graph Diffusion Network surrogate on a chosen ABM configuration. Use the following command-line options:

```bash
python train_model.py \
  --abm_model   <predatorprey|schelling> \
  --model_type  <surrogate|ablation> \
  --parameter   <psi1|psi2|psi3|psi4> \      # for predatorprey
               <xi1|xi2|xi3> \               # for schelling
  --learning_rate  <float> \
  --T_diffusion    <int> \
  --n_epochs       <int>
```

**Example:**

Train the surrogate on the Predator–Prey model with parameter psi2, a learning rate of 1e-4, 50 diffusion steps, and 200 epochs:

```bash
python train_model.py \
  --abm_model predatorprey \
  --model_type surrogate \
  --parameter psi1 \
  --learning_rate 1e-5 \
  --T_diffusion 100 \
  --n_epochs 100
```

Use -h or --help to display all available options and their defaults:

```bash
python train_model.py --help
```

3.	Evaluate the Trained Surrogates

After training, assess model fidelity on held-out trajectories using the provided evaluation scripts.

```bash
python evaluate_predator_prey.py \
  --model_type <surrogate|ablation> \
  --psi        <psi1|psi2|psi3|psi4>
```

For example, to evaluate the surrogate model trained with psi1:

```bash
python evaluate_predator_prey.py --model_type surrogate --psi psi1
```

#### Notes

For further assistance, please refer to the comments in the scripts or notebooks.

