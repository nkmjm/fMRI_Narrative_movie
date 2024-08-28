# fMRI_Narrative_movie
The demonstrations are based on the following paper:<br>
Hiroto Q. Yamaguchi, Naoko Koide-Majima, Rieko Kubo, Tomoya Nakai, Shinji Nishimoto<br>
"Multi-Annotation fMRI Dataset from Extensive Drama and Film Viewing for Perceptual and Cognitive Research."<br>
[URL](xxx)

## Initial setup
### Prepare data used in the demonstrations
- Data is available from [OpenNeuro](url).
- In the demonstrations, subdirectories under "derivative" are used.
### Path management and configuration updates
- To set the root directory of the data, specify the root directory as "dir: derivative:" in the config file ("config__drama_data.yaml") under the "util."
- To visualize data on the flattened cortical map, set the pycortex filestore database directory according to the instructions at https://gallantlab.org/pycortex/auto_examples/quickstart/show_config.html.

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


