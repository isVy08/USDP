import numpy as np
from tqdm import tqdm
from decoder.helper import *
from decoder.scorer import * 
from utils import scale, write_pickle
from decoder.candidate import Candidate


def sample(prev_seq, prev_token, lookup):
        
    indices = [token.i for token in prev_seq if not isinstance(token, str)]
    
    if isinstance(prev_token, str): 
        return resample(prev_seq, indices, lookup)
    
    samples = []
    for token in lookup[prev_token]:
        if token.i not in indices:
            if token.text in ('no', 'not', 'never', 'none', 'neither', 'không', 'chưa') or 'obj' in token.dep_:
                return [token]
            samples.append(token) 
    
    if len(samples) == 0:
        # Trace back to the latest <s>
        i = -1
        complete = False
        while i >= -len(prev_seq) and not isinstance(prev_seq[i], str):
            if prev_seq[i].pos_ == 'VERB':
                complete = True
                break
            i -= 1
            
        if complete:
            samples.append('<s>') # induce splitting if a complete sentence is formed
        else:
            return resample(prev_seq, indices, lookup) 
   
    return samples

def resample(prev_seq, indices, lookup):
    
    for token in prev_seq:
        # back to the nearest-to-root token not been sampled
        if not isinstance(token, str):
            choices = [tok for tok in lookup[token] if tok.i not in indices] 
            if len(choices) > 0:
                return choices 

def top_k_sampling(beams, k, W):
    
    beam_fluency = np.array([beam.fluency for beam in beams])
    beam_fluency = scale(beam_fluency, 0, 1)

    n = len(beams)
    beam_score = np.array([beams[i].similarity + W * beam_fluency[i] + beams[i].depth for i in range(n)])
    _, beam_indices = np.unique(beam_score, return_index=True)

    if k == 1:
        return beams[beam_indices[-1]]

    while len(beam_indices) < k:
        beam_indices = np.tile(beam_indices, 2)
    
    return [beams[i] for i in beam_indices[-k:]]
         
def simplify(source_data, nlp, evaluator, critic, W, 
            beam_width, length_ratio, min_similarity, output_path=None, stat_path=None, lang='en'):
   
    batch_size = len(source_data)
    global decoded
    decoded = [None for _ in range(batch_size)]
    stats = [None for _ in range(batch_size)]
    
    batch = tqdm(range(batch_size)) 
    for ba in batch:

        # Initiate beams with the subject
        input_text = source_data[ba][:-1]
        
        doc = nlp(input_text)
        head = get_root_head(doc, lang)
        tree, tree_depth = get_tree_depth(doc)
        
        lookup = get_relations(doc, head[-1])
        L = len(doc)

        max_length = int(L * length_ratio)
 
        incomplete  = []
        for _ in range(beam_width):
            candidate = Candidate(None, head[-1], 1)
            candidate.sequence = head
            incomplete.append(candidate)

        step = tqdm(range(L)) 
        for t in step: 

            # processing beam
            new_beams = []
            for be in range(beam_width):
                step.set_description(f'Step {t} : Batch {ba} - Beam {be}')
                # print('Processing beam', be)
                prev_node = incomplete[be]
                prev_seq = prev_node.sequence
                prev_token = prev_node.token

                samples = sample(prev_seq, prev_token, lookup)

                if samples:

                    for token in samples:
                        s, f, d, text, ordered_seq = score(prev_seq, token, input_text, evaluator, critic, tree)
                        
                        stop = prev_node.length + 1 >= max_length and s >= min_similarity
                        if isinstance(token, str) and stop: 
                            step.set_description(f'Done batch {ba}') 
                            ordered_seq.append(token)
                            decoded[ba] = split(ordered_seq)
                            stats[ba] = (s, f, d, prev_node.length + 1, tree_depth, L)
                            break
                        
                        length = prev_node.length if isinstance(token, str) else prev_node.length + 1
                        beam = Candidate(prev_node, token, length)
                        beam.similarity = s
                        beam.fluency = f
                        beam.depth = 1/d
                        beam.sequence = ordered_seq
                        beam.text = text
                        new_beams.append(beam)
            
                if decoded[ba]:
                    break
            
            if decoded[ba] or len(new_beams) == 0:
                break

            incomplete = top_k_sampling(new_beams, beam_width, W)
        
        if decoded[ba] is None:
            node = top_k_sampling(incomplete, 1, W)
            if not isinstance(node.token, str):
                node.sequence.append('<s>')
            decoded[ba] = split(node.sequence)
            stats[ba] = (node.similarity, node.fluency, 1/node.depth, node.length, tree_depth, L)
        
        if output_path:
            write_pickle(decoded, output_path)
        
        if stat_path:
            write_pickle(stats, stat_path)

    return decoded, stats
