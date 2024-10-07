# fMRI_Narrative_movie
This code is prepared for analyzing [an fMRI dataset](https://openneuro.org/datasets/ds005531), which was collected to investigate brain representations related to cognitive functions involved in narrative understanding. The overview of the data is as follows.<br>
- We measured the BOLD responses during 6 participants watched 9 titles of narrative movies (8.86 hours in total).
- For the movies, there are three types of annotations related to narrative understanding (objective information, speech transcription, narrative content (story)).

This code follows the analytical procedures outlined in the following paper.
- Hiroto Q. Yamaguchi, Naoko Koide-Majima, Rieko Kubo, Tomoya Nakai, Shinji Nishimoto<br>
  "Multi-Annotation fMRI Dataset from Extensive Drama and Film Viewing for Perceptual and Cognitive Research."<br>
  [URL](xxx)

We have prepared three types of demonstration codes related to these analyses:<br>
- "demo__feature_extruction.ipynb"
- "demo__encoding_model_fitting.ipynb"
- "demo__pycortex_visualization.ipynb"

## Initial setup
### Prepare data used in the demonstrations
- Data are available from [OpenNeuro](https://openneuro.org/datasets/ds005531).
- In the demonstrations, subdirectories under "derivative" are used.
- The subdirectories includes
  - *preprocessed_data*: Preprocessed fMRI BOLD responses.　
  - *annotation*: Three types of annotations.
  - *feature*: LLM latent features extracted from the annotations.
  - *pycortex_db*: Pycortex database, which is used for visualization of the analytical results on cortical surface.
  - *localizer*: Statistics and estimates derived from 9 types of functional localizer data.
  - *accuracy*: Prediction accuracies of an individual encoding model using the latent features for the three types of annotations.
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
"demo__feature_extruction.ipynb"
- Extracting LLM features ([GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)) from sentences attributed to each 1-second movie scene of each type of the annotations.
- Parameters
  - *model_lang*: Language type of the GPT-2 models (Japanese (jp)/ English (en)).
  - *layer_no*: The number of layers to extract the latent features (1–24).
  - *text_path*: The file paths of the annotations (text data).

"demo__encoding_model_fitting.ipynb"
- Constructing individual　encoding models to predict the brain responses from the LLM latent features.
- Parameters
  - *subject_name*: The name of the participant (e.g., "sub-S03").
  - *feat_names*: Features to predict the BOLD responses (e.g., "")
  - *movtitle_test*: The movie title used in the model testing (e.g., "bigbangtheory").

"demo__pycortex_visualization.ipynb"
- Visualization of results (e.g. prediction accuracies of an encoding model) on the cortical surface using [Pycortex](https://gallantlab.org/pycortex/index.html).
- Using this demonstration codes, you can visualize statistics and estimates regarding the 9 functional localizers and the encoding accuracies.
- Parameters
  - *subject_name*: The name of the participant (e.g., "sub-S03")
  - *stat_path*: The file path of the statistics or estimates.

## Important notes
"util/util_ridge.py" includes an excerpt from Gallant lab's Github code.<br>
"generate_leave_one_run_out" You can find it at: <br>https://github.com/gallantlab/voxelwise_tutorials/blob/main/voxelwise_tutorials/utils.py

