import argparse
from utils import load, load_pickle, write_pickle
from decoder.simplifier import *

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="The config file specifying all params.")
    params = parser.parse_args()
    with open(params.config) as f:
        config = json.load(f)
    n = Namespace()
    n.__dict__.update(config)
    return n

def run_simplification(config): 

    # Load data
    source_data = load(config.dataset_path)
    source_data = source_data[config.start : config.end]

    # Load objects
    if config.lang == 'en':
        import spacy
        import benepar
        nlp = load_pickle(config.nlp_path)
    else: 
        from decoder.vnnlp import pos_converter, dep_converter, Token, nlp

    evaluator = load_pickle(config.evaluator_path) 
    critic = load_pickle(config.critic_path)

    beam_width, length_ratio = config.beam_width, config.length_ratio 
    W, min_similarity = config.weight, config.min_similarity

    decoded, stats = simplify(source_data, nlp, evaluator, critic, W, 
                        beam_width, length_ratio, min_similarity, 
                        config.output_path, config.stat_path, config.lang)
    
    write_pickle(decoded, config.output_path)
    if config.stat_path:
        write_pickle(stats, config.stat_path)
   


if __name__ == '__main__':
    config = get_params()
    run_simplification(config)
    print('Done')