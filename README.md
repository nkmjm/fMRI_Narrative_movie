# fMRI_Narrative_movie
The demonstrations are based on the paper:

# Initial set up
## Prepare data used in the demonstrations. 
- Data is available from OpenNeuro <url>.
- In the demonstrations, subdirectories under "derivative" are used.
## Path Management and Configuration Updates
- Need to set a root dir of the config file () under "util."
  In the {xxx} of the config file, specify the path where the directory "derivative" is stored.
- To visualize data on the flattened cortical map, need to set pycortex filestore<https://gallantlab.org/pycortex/auto_examples/quickstart/show_config.html> to the database root_dir.

# 

- git clone https://github.com/gallantlab/pycortex.git cd pycortex; python setup.py install

