# USDP
Codes for reproducing experiments in the paper [Unsupervised Sentence Simplification via Dependency Parsing](https://arxiv.org/pdf/2206.12261)

## Requirements 
Python 3.6 or 3.7 is required. 
```
cd USDP
pip install -r requirements
```

## Model Evaluator
The pre-trained models used in this experiment include:
* Spacy + Benepar Parsing: `nlp.pickle` 
* SBERT sentence embeddings:
  * Monolingual `paraphrase-mpnet-base-v2`:  `evaluator.pickle`
  * Multilingual `distiluse-base-multilingual-cased-v2`: `mtlevaluator.pickle`
* Constituent-based 4-gram Kneser-Ney smoothing
  * English: `critic.pickle`
  * Vietnamese: `vncritic.pickle`   

### 1. Spacy model 
`nlp.pickle` is Spacy object for NLP parsing. It can be directly obtained by installing Spacy and calling the object 
```
pip install -U spacy
python -m spacy download en_core_web_sm

python
import spacy
from utils import write_pickle
nlp = spacy.load("en_core_web_sm")
write_pickle(nlp, 'nlp.pickle')
```

### 2. SBERT model 
The pre-trained SBERT models are available [here](https://www.sbert.net/docs/pretrained_models.html)

```
python
from sentence_transformers import SentenceTransformer
from utils import write_pickle
model = SentenceTransformer('paraphrase-mpnet-base-v2')
write_pickle(model, 'evaluator.pickle')
```

if `paraphrase-mpnet-base-v2` is no longer avaiable, try `all-mpnet-base-v2`.

### 3. Kneser-Ney smoothing model
To train the English Fluency model `critic.pickle`, 

#### Step 1: Parsing
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
#### Step 2: Training
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

## References 
If you use the codes or datasets in this repository, please cite our paper 
```
@article{vo2022unsupervised,
  title={Unsupervised Sentence Simplification via Dependency Parsing},
  author={Vo, Vy and Wang, Weiqing and Buntine, Wray},
  journal={arXiv preprint arXiv:2206.12261},
  year={2022}
}
```

