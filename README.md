# RAMP starting kit on the prediction of subventions allocated to projects in Paris

Here we propose a challenge, that is to predict the subventions allocated by the Parisian townhouse to a given project. The description of the challenge and the data is available in the notebook. The data are available at https://opendata.paris.fr/explore/dataset/subventions-accordees-et-refusees.

In the submission folder, we propose a baseline for the challenge. The baseline is compatible with ramp servers.

## Dependencies <TODO>

This starting kit requires Python and the following dependencies:

* `numpy`
* `scipy`
* `pandas`
* `scikit-learn`
* `matplolib`
* `seaborn`
* `jupyter`
* `ramp-workflow`

## Important Files

We include in our repository the following files to work on the challenge.

* [PSP_starting_kit.ipynb](PSP_starting_kit.ipynb): To get started with the challenge. Find descriptions, graphs and basic pre-processing of the features. Use the following command from the root directory to run it:
  
  ```bash
  $ jupyter-notebook PSP_starting_kit.ipynb
  ```

* [download_data.py](download_data.py): Use this python script to download the dataset for this challenge. By default, it will store the files on the directory [data](data).
* [submissions](submissions): This directory contains all the directories (e.g. [starting_kit](starting_kit)) used for the local submissions. Each of them represent an individual submission and must contain two files
  
  * `feature_extractor.py`: Implementation of the class `FeatureExtractor` for the preprocessing of the features.
  * `regressor.py`: Implemenation of the class `Regressor` for the training of the model and the prediction.
  
* <TODO> To complete.

## Getting start with RAMP

Before testing locally using **RAMP**, please install the `ramp-workflow` using the following command

 ```bash
 $ pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git
 ```

Then, to test locally your model, please use the following command replacing `starting_kit` with the name of the directory containing the python scripts for the submission.

```bash
ramp_test_submission --submission starting_kit
```

**Note:** Calling just the command `$ ramp_test_submission` will work aswell, but it will use the `starting_kit` directory by default.

For more information on the [RAMP](http:www.ramp.studio) ecosystem go to
[`ramp-worflow`](https://github.com/paris-saclay-cds/ramp-workflow).