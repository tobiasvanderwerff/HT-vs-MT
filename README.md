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


## Future work

There are several directions that this research can still be taken in the future.
Below is an overview of potentially interesting things to explore, and how the code
should be modified to make it work.

- [Other source languages](#other-source-languages)
- [Including the source text](#including-the-source-text)
- [3-way classification](#3-way-classification)
- [Other domains](#other-domains)
- [Adding translationese-inspired features](#adding-translationese-inspired-features)


### Other source languages

#### What
Can we recognize translations from a different source language?

So far we've been using German -> English translations, but English and German are
relatively similar. Perhaps the effect of MT on translations could be more noticeable in
the case of two languages that are significantly different, e.g. from two different
language families.

#### How
This only requires changing the data that the classifier is trained/evaluated on. The
`--root_dir` argument specifies the root of your data directory, which should contain a
folder named `data`, with the following structure:

```
.
|- data
    |- deepl
         |- train
         |- dev
         |- test
    |- google
         |- train
         |- dev
         |- test
```

Using translations from a different source language should be as simple as creating
a new data directory and placing the translations in their appropriate folders,
according to the structure shown above. The `experiments` folder shows various
examples of this.

One thing that is important to note is that your text files should have specific
file extensions that indicate their identity. These are the following:

```
*.txt:       Original text / human translations
*.deepl.en:  DeepL translations
*.en.google: Google Translate translations
*.wmt:       WMT submission translations
```

I guess these file extensions could be a bit more consistent, but right now this is what
is used to identify the different kinds of translations (see `data.py:35`). The
following naming conventions are used for the full file names. Using the example of
WMT14:

```
# Original
org_de_wmt14.txt:                           original German
org_de_wmt14.{deepl.en, en.google, wmt}:    MT of original German
trans_en_wmt14.txt:                         HT of original German

# Translationese
trans_de_wmt14.txt:                         translationese German
trans_de_wmt14.{deepl.en, en.google, wmt}:  MT of translationese German
```

Note that there are no human translations of translationese German (these are not
included in WMT).

### Including the source text

#### What
Does including the source text help classification?

#### How
Effectively, this implies classifying sentence pairs, whereas so far we have been
classifying single sentences (or documents). Since the source and translation are of
two different languages (e.g. German and English), this requires a multilingual
language model. At the time of writing, Roberta-XLM is a state-of-the-art multilingual
language model, which we tried some experiments with. In `models.py`, you can find a
class named
`BilingualSentenceClassifier` that classifies MT vs. HT based on two-sentence inputs.
This will probably be a good starting point for experimenting with bilingual
classification. So far, the results were a bit mixed.

Using sentence pairs also means that the data folder will look different. We store the
data files in the same format as before (one sentence per line), but now include an
additional file containing the source sentences, for each translation file. The file
name conventions are as follows:

```
|-------------------------------|----------------------------------|
| source file name              | translation file name            |
|-------------------------------|----------------------------------|
| org_de_wmt{wmt_year}.txt      | trans_en_wmt{wmt_year}.txt       |
| (original text)               | org_de_wmt{wmt_year}.deepl.en    |
|                               | org_de_wmt{wmt_year}.en.google   |
|                               | org_de_wmt{wmt_year}.wmt         |
|-------------------------------|----------------------------------|
| trans_de_wmt{wmt_year}.txt    | trans_de_wmt{wmt_year}.deepl.en  |
| (translationese)              | trans_de_wmt{wmt_year}.en.google |
|                               | trans_de_wmt{wmt_year}.wmt       |
|-------------------------------|----------------------------------|

```

Given the source file name on the left of the table, the corresponding translation
should be named according to one of the file names on the right. The source and
corresponding translation file should be in the same folder.

Whereas for monolingual sentences we would call `load_corpus()` from `data.py`, we now
call
`load_corpus_sentence_pairs()` instead. This can be done by passing
the `--load_sentence_pairs` argument, which will load sentence pairs and use Roberta-XLM
to classify them.

### 3-way classification

#### What
Right now, the code is set up to do 2-way classification, i.e. MT vs. HT. Potentially
interesting is to try 3-way classification: MT vs. HT vs. original. Original then refers
to a text that has not been translated.

#### How
This would require additional code that loads in the original texts along with the MT
and HT data. This should be relatively straightforward, by adding a key for the third
class ('2') in `data.py:35`, where the original texts can be loaded.

Huggingface makes it easy to change the number of output classes for a language
model classifier. In `classifier_trf_hf.py:82`, simply change the `num_labels` argument
from 2 to 3.



### Other domains

#### What
We have not yet tested how well the classifiers would work across domains. WMT contains
news stories, but it is not clear how well the approach would work for text originating
from other domains, e.g. text scraped from the web.

#### How
This requires changing the data. See the previous explanation [above](#other-source-languages).



### Adding translationese-inspired features

#### What
It may be interesting/useful to combine handcrafted features for detecting
translationese with learned features from a pretrained LM (although there is no
direct indication that this should help). There exists various literature on how
such features can be designed. (An interesting resource may be the master thesis by
Yu-Wen Chen, titled "Automatic Detection of Different Types of Translation Based on
Translationese Features".)

#### How
This would first require figuring out how the translationese features could be
combined with the learned features (i.e. where are they combined in the processing
pipeline). It would require additional code for extracting the handcrafted features
from the data, and combining it with the existing Huggingface code.
