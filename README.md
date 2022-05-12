# USDP
Codes for reproducing experiments in the paper [Unsupervised Sentence Simplification via Dependency Parsing](https://isvy08.github.io/USDP.pdf).

## Requirements 
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
* Run this command-line for constituent parsing so that each line now becomes a sequence of constituents. Pickle the data as `train_ngram.data`. 

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

Note that the outputs of `RM+EX+LS+RO` on `PWKP` are created by reproducing the experiment from [Edit-Unsup-TS](https://github.com/ddhruvkr/Edit-Unsup-TS)

## Running USDP
### Phase 1: Structural Simplification
To reproduce `USDP-Base` on English data, simply run
```
python run_generation.py evaluation/config_en_base.json
```

Change the path to `evaluation/config_vn_base.json` for Vietnamese simplification. Feel free to modify the parameters to experiment with other variants, such as `USDP-Match`.

### Phase 2: Back Translation
Successfully completing phase 1 will output sentences that are structurally simpler than the original ones. You can further implement lexical simplification and paraphrasing by back-translating the outputs using any multilingual pre-trained machine translation system. We simply make use of [Google Translate service](http://translate.google.com) in our experiment.    




