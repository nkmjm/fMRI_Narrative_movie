# fMRI_Narrative_movie
This code is prepared for analyzing [an fMRI dataset](openNeuroURL)(the overview of the data is as follows), which was collected to investigate brain representations related to cognitive functions involved in narrative understanding.<br>
- We measured the BOLD responses during 6 participants watched 9 titles of narrative movies (8.6 hours in total).
- For the movies, there are three types of annotations related to narrative understanding (objective information, speech transcription, narrative content (story)).<br>
This code follows the analytical procedures outlined in the following paper.<br>
We have prepared three types of demonstrations related to this analysis.<br>
<br>
Hiroto Q. Yamaguchi, Naoko Koide-Majima, Rieko Kubo, Tomoya Nakai, Shinji Nishimoto<br>
"Multi-Annotation fMRI Dataset from Extensive Drama and Film Viewing for Perceptual and Cognitive Research."<br>
[URL](xxx)<br>

## Initial setup
### Prepare data used in the demonstrations
- Data is available from [OpenNeuro](url).
- In the demonstrations, subdirectories under "derivative" are used.
- The subdirectories includes
  - *preprocessed_data*: Preprocessed fMRI bold responses.　
  - *annotation*: Three types of annotations.
  - *feature*: LLM latent features extracted from the annotations
  - *pycortex_db*: Pycortex database, which is used for visualization of the analytical results on cortical surface.
  - *localizer*: Statistics and estimates derived from 9 types of functional localizer data.
  - *ridge*: Prediction accuracies of an individual encoding model using the latent features for the three types of annotations.
  LLM latent features extracted from the annotations, pycortex database, localizer information, and results of encoding model fitting.
### Path management and configuration updates
- To set the root directory of the data, specify the root directory as "dir: derivative:" in the config file ("config__drama_data.yaml") under the "util" directory.
- To visualize data on the flattened cortical map, set the pycortex filestore database directory in the pycortex config file (~/.config/pycortex/options.cfg).
``filestore = {your pycortex_db dir}``
- More detailed instructions can be found at https://gallantlab.org/pycortex/auto_examples/quickstart/show_config.html.

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

## Demonstration
Start jupyter notebook and setting Kernel to {env_name}.
- "demo__feature_extruction.ipynb"
  - Extracting LLM features ([GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)) from sentences attributed to each 1-second movie scene of each type of the annotations.
- "demo__encoding_model_fitting.ipynb"
  - Constructing encoding model to predict the brain responses from the LLM latent features.
- "demo__pycortex_visualization.ipynb"
  - Visualization of results (e.g. prediction accuracies of an encoding model) on the cortical surface using [Pycortex](https://gallantlab.org/pycortex/index.html).
  - Using this demonstration codes, you can visualize statistics and estimates regarding the 9 functional localizers and the encoding accuracies.


