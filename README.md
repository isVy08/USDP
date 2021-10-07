# USPD
Unsupervised Sentence Simplification via Dependency Parsing

## Requirments 
Python 3.6 or 3.7 is required. 
```
cd USDP
pip install -r requirements
```

## Pre-trained models
Download pre-trained models into working directory from [this collection](https://drive.google.com/drive/folders/1YLUlufs5g77QyzjP1yW4c54PIa9KwwXS?usp=sharing)
* Spacy + Benepar Parsing: `nlp.pickle`
* SBERT sentence embeddings:
  * Monolingual `paraphrase-mpnet-base-v2 `:  `evaluator.pickle`
  * Multilingual `distiluse-base-multilingual-cased-v2`: `mtlevaluator.pickle`
* Constituent-based 4-gram Kneser-Ney smoothing:
  * English: `critic.pickle`
  * Vietnamese: `vncritic.pickle`   

## Fluency model
If you wish to train your own Fluency model, instead of using `critic.pickle` or `vncritic.pickle`

### Step 1: Parsing
* Obtain any corpus (~1M sentences) wherein each line is a sentence
* Run this command-line to parse each line so that each line is a sequence of constituents, and pickle the data as `train_ngram.data`. 

  * English:
  ```
  python run_ngram.py your-data-path train_ngram.data parse en
  ```
  
  * Vietnamese:
  ```
  python run_ngram.py your-data-path train_ngram.data parse vi
  ```
### Step 2: Training
Run this command-line to train a 4-gram KNS on `train_ngram.data`
```
python run_ngram.py train_ngram.data critic.pickle train
```

## Data
The `data` folder in `evaluation` contains the testing data of
* TurkCorpus: `turkcorpus.orig`
* PWKP: `pwkp.test.orig`
* CP_Vietnamese-VLC (extracted): `vndata.orig`

TurkCorpus and PWKP have their ground-truth references with extension `.simp` and outputs of competing models in a corresponding folder. All data is gratefully borrowed from [EASSE](https://github.com/feralvam/easse) and [Under the Sea NLP](https://github.com/undertheseanlp/resources/tree/master/resources/CP_Vietnamese-VLC).

## Running USDP
To reproduce `USDP-base` on English data, simply run
```
python run_generation.py evaluation/config_en_base.json
```

Change the path to `evaluation/config_vn_base.json` for Vietnamese simplification. Feel free to modify the parameters to experiment with other variants, such as `USDP-Match`.




