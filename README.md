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
- Create a new environment and install basic packages<br>
```
conda create --name {env_name} python=xx.xx<br>
conda activate {env_name}``<br>
conda install os numpy pickle scipy yaml matplotlib h5py jupyter　sklearn itertools statsmodels random
```
- Install himalaya (for "")<br>
- Install torch (for "demo__feature_extraction.ipynb")<br>
- Install pycortex (for "demo__pycortex_visualization.ipynb")<br>
```
git clone https://github.com/gallantlab/pycortex.git<br>
cd pycortex; python setup.py install
```

