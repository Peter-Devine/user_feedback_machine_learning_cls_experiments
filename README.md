# About
This repository is based on the user feedback classification paper from Peter Devine, Yun Sing Koh, and Kelly Blincoe.

The code for this paper with written by Peter Devine.
If you would like to run this code yourself, or would like to use it for something else, please feel free to.
If you have any problems with running the code, feel free to reach out and create an issue. I will be happy to help.


# How to run

* Install Python 3
* Download PyTorch using the instructions on https://pytorch.org/
(E.g. `pip3 install torch torchvision torchaudio`)
* Pip install the following packages: `pip install -U scikit-learn pandas scipy transformers`
* Run `run.py`
(I.e. `python run.py`)
* Install jupyter notebook from https://jupyter.org/install
(E.g. `pip install jupyterlab`)
* Run the cells in `Results.ipynb`

N.B. This repository comes with the results already included in `results/`, but you will need to train the models again if you want to fully replicate the study.

If you are looking for a classifier to classify user feedback into bug reports and feature requests, please see the models at:
https://huggingface.co/Peterard/distilbert_bug_classifier
https://huggingface.co/Peterard/distilbert_feature_classifier
These pages have full instructions for how to run the models.

(The `classifier_upload.ipynb` notebook was how I uploaded these models.)