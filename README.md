# MT vs. HT
All code, data, experiment scripts and results for the EAMT 2022 paper "Automatic
Discrimination of Human and Neural Machine Translation: A Study with Multiple
Pre-Trained Models and Longer Context".

Most scripts for running experiments are written for the SLURM workload manager, which
is used on our local High Performance Cluster. For the most part, these are simply bash
scripts with some additional SLURM-specific parameters defined at the top of the script.

## How to run
The `classifier_trf_hf.py` script is the main entry point for training a classifier
using a pretrained language model. A SVM classifier can be trained with the
`classifier_svm.py` script. The data is provided per experiment, and a full list of all
the experiments can be found in `experiments/experiments.yaml`. By default, the data
from WMT 2008-2019 is used, without Translationese texts.

In order to train a classifier, first install the dependencies:

```shell
python -m venv env  # create a new virtual environment
source env/bin/activate  # activate the environment
pip install -r requirements.txt  # install the dependencies
```

Then, run the main script. For example:

```shell
python classifier_trf_hf.py --arch microsoft/deberta-v3-large --learning_rate 1e-5 --batch_size 32
```

Many more arguments can be passed than those shown above; if you want to see a list of
all possible arguments, run

```shell
python classifier_trf_hf.py -h
```
