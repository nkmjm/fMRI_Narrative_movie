# fMRI_Narrative_movie
The demonstrations are based on the paper:

## Initial set up
### Prepare data used in the demonstrations. 
- Data is available from OpenNeuro <url>.
- In the demonstrations, subdirectories under "derivative" are used.
### Path Management and Configuration Updates
- Need to set a root dir of the config file () under "util."
  In the {xxx} of the config file, specify the path where the directory "derivative" is stored.
- To visualize data on the flattened cortical map, need to set pycortex filestore<https://gallantlab.org/pycortex/auto_examples/quickstart/show_config.html> to the database root_dir.

## Environment setup
Assuming the use of Miniconda.
- Create a new environment and install basic packages.<br>
```
conda create --name {env_name} python=3.10
conda activate {env_name}
conda install os numpy pickle scipy yaml matplotlib h5py jupyter　sklearn itertools statsmodels random
```
- Install [himalaya](https://github.com/gallantlab/himalaya).<br>
for "demo__dncoding_model_fitting.ipynb"
```
pip install himalaya
```
- Install PyTorch according to the instructions on the [PyTorch website](https://pytorch.org/).<br>
for "demo__feature_extraction.ipynb"
- Install pycortex according to the instruction on the [Pycortex website](https://gallantlab.org/pycortex/install.html).<br> (for "demo__pycortex_visualization.ipynb")<br>


