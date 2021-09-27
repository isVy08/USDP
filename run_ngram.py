import sys
from tqdm import tqdm
from utils import load, load_pickle, write_pickle

def parse(sent, nlp):
    doc = nlp(sent)
    return [token.pos_.lower() for token in doc] 

def genrate_data(corpus, nlp, saved_path):
    n = len(corpus)
    pbar = tqdm(range(n))
    data = []
    for i in pbar:
        out = parse(corpus[i], nlp)
        data.append(out)
        if i % 100 == 0: # save every 100 samples
            write_pickle(data, saved_path)
    write_pickle(data, saved_path) 

def train(parsed_data, saved_path, ngram=4):
    from nltk.lm import KneserNeyInterpolated
    from nltk.lm.preprocessing import padded_everygram_pipeline
    print('Start training:')
    train_data, padded_sents = padded_everygram_pipeline(ngram, parsed_data)
    model = KneserNeyInterpolated(ngram)
    model.fit(train_data, padded_sents)
    write_pickle(model, saved_path)


if __name__ == "__main__":

    data_path = sys.argv[1]
    saved_path = sys.argv[2]
    
    if sys.argv[3] == 'parse':
        assert sys.argv[4] in ('vi', 'en'), 'Language not currently supported'
        if sys.argv[4] == 'vi':
            from decoder.vnnlp import *
        else:
            import spacy
            import benepar
            nlp = load_pickle('decoder/models/nlp.pickle')
        
        corpus = load(data_path)
        genrate_data(corpus, nlp, saved_path)
    
    elif sys.argv[3] == 'train':
        parsed_data = load_pickle(data_path)
        train(parsed_data, saved_path)

    print('Done')

