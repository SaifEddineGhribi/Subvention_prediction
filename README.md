# RAMP starting kit on the prediction of grants allocated to projects in Paris


**Authors:**
**Angelo CANESSO - Enrique GOMEZ - Saif GHRIBI**
**Ramzi HAMDI - Ahlem JOUIDI - Leandro NASCIMIENTO**

Associations are an important part of French population's lives. In fact, one half of the French over 18 years old takes part of at least one association. Either cultural, educative of sportive, an association needs funds and one of the most repanded ways of get them is by applying for public grants.

Here we propose a challenge, that is to predict the grants allocated by Paris to associations. The description of the challenge is available in the notebook. The data and its description is available at https://opendata.paris.fr/explore/dataset/subventions-accordees-et-refusees.

## Dependencies

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

* [Data_Camp_project.ipynb](Data_Camp_project.ipynb): To get started with the challenge. Find descriptions, graphs and basic pre-processing of the features. Use the following command from the root directory to run it:
  
  ```bash
  $ jupyter-notebook Data_Camp_project.ipynb
  ```

* [submissions](submissions): This directory contains all the directories (e.g. [starting_kit](starting_kit)) used for the local submissions. Each of them represent an individual submission and must contain two files
  
  * `feature_extractor_clf.py`: Implementation of the class `FeatureExtractor` for the preprocessing of the features for the classification.
  * `feature_extractor_reg.py`: Implementation of the class `FeatureExtractor` for the preprocessing of the features for the regression.
  * `regressor.py`: Implementation of the class `Regressor` for the training of the model and the prediction.
  * `classifier.py`: Implementation of the class `Classifier` for the training of the model and the prediction.
  * `problem.py`: Definition of the problem for the RAMP server, it the loss and all the necessary functions for the CV of RAMP.

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
